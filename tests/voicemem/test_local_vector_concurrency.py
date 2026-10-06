"""Local vector consistency and lock scope using synthetic, temporary stores."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import inspect
import os
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from voicemem.leftbrain.mem0_backend_store import (
    Mem0BackendStore, _MEM0_CLIENT_CACHE, _MEM0_CLIENT_CACHE_LOCK,
    _synchronize_local_qdrant,
)

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    QdrantClient = None

try:
    import mem0
except ImportError:
    mem0 = None


class LocalClientLockTests(unittest.TestCase):
    def test_nested_sdk_calls_remain_reentrant_and_exceptions_release_lock(self):
        class Client:
            def query_points(self, *, fail=False):
                if fail:
                    raise ValueError('synthetic failure')
                return self.count()

            def count(self):
                return 3

        client = Client()
        signature = inspect.signature(client.query_points)
        _synchronize_local_qdrant(client)
        self.assertEqual(inspect.signature(client.query_points), signature)
        with self.assertRaisesRegex(ValueError, 'synthetic failure'):
            client.query_points(fail=True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.assertEqual(pool.submit(client.query_points).result(timeout=2), 3)

    def test_a_blocked_client_does_not_block_another_space(self):
        entered, release = threading.Event(), threading.Event()

        class Client:
            def upsert(self):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError('test writer was not released')

            def query_points(self):
                return 'independent space'

        first, second = Client(), Client()
        _synchronize_local_qdrant(first)
        _synchronize_local_qdrant(second)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(first.upsert)
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(pool.submit(second.query_points).result(timeout=1),
                                 'independent space')
            finally:
                release.set()
            writer.result(timeout=2)


@unittest.skipIf(QdrantClient is None, 'requires the runtime qdrant-client dependency')
class NativeVectorConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.client = QdrantClient(':memory:')
        _synchronize_local_qdrant(self.client)
        self.client.create_collection('fixture', vectors_config=models.VectorParams(
            size=16, distance=models.Distance.COSINE))
        self.client.upsert('fixture', [self.point() for _ in range(188)])

    def tearDown(self):
        self.client.close()

    def point(self):
        return models.PointStruct(id=str(uuid.uuid4()), vector=[0.1] * 16,
                                  payload={'user_id': 'fixture', 'nested': {'value': 1}})

    def query(self):
        return self.client.query_points('fixture', query=[0.1] * 16, limit=20,
            query_filter=models.Filter(must=[models.FieldCondition(
                key='user_id', match=models.MatchValue(value='fixture'))]))

    def test_write_cannot_resize_arrays_during_an_active_query(self):
        entered, release, write_started = (threading.Event() for _ in range(3))
        collection = self.client._client.collections['fixture']
        calculate = collection._payload_and_non_deleted_mask

        def pause_query(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError('test query was not released')
            return calculate(*args, **kwargs)

        def write():
            write_started.set()
            self.client.upsert('fixture', [self.point()])

        with patch.object(collection, '_payload_and_non_deleted_mask', pause_query):
            with ThreadPoolExecutor(max_workers=2) as pool:
                query = pool.submit(self.query)
                try:
                    self.assertTrue(entered.wait(1))
                    writer = pool.submit(write)
                    self.assertTrue(write_started.wait(1))
                    with self.assertRaises(FutureTimeoutError):
                        writer.result(timeout=.05)
                    self.assertEqual(len(collection.ids), 188)
                finally:
                    release.set()
                self.assertEqual(len(query.result(timeout=2).points), 20)
                writer.result(timeout=2)
        self.assertEqual(self.client.count('fixture').count, 189)

    def test_concurrent_insert_update_delete_search_batch_and_scroll(self):
        barrier = threading.Barrier(5)

        def writer():
            barrier.wait(timeout=2)
            for _ in range(100):
                point = self.point()
                self.client.upsert('fixture', [point])
                self.client.set_payload('fixture', {'nested': {'value': 2}}, [point.id])
                self.client.delete('fixture', [point.id])

        def reader():
            barrier.wait(timeout=2)
            for _ in range(60):
                self.assertEqual(len(self.query().points), 20)
                batch = self.client.query_batch_points('fixture', requests=[
                    models.QueryRequest(query=[0.1] * 16, limit=5, with_payload=True)])
                self.assertEqual(len(batch[0].points), 5)
                records, _ = self.client.scroll('fixture', limit=20)
                self.assertEqual(len(records), 20)

        with ThreadPoolExecutor(max_workers=5) as pool:
            tasks = [pool.submit(writer)] + [pool.submit(reader) for _ in range(4)]
            for task in tasks:
                task.result(timeout=10)
        self.assertEqual(self.client.count('fixture').count, 188)
        snapshot = self.query().points[0]
        self.client.set_payload('fixture', {'nested': {'value': 9}}, [snapshot.id])
        self.assertEqual(snapshot.payload['nested']['value'], 1)


@unittest.skipIf(mem0 is None or QdrantClient is None, 'requires runtime mem0ai and qdrant-client')
class Mem0LockScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'fixture'
        self.root.mkdir()
        self.entered, self.release = threading.Event(), threading.Event()

        outer = self

        class Embedder:
            dimensions = 16
            model_name = 'synthetic-vectors'

            def embed_texts(self, texts):
                if any(text == 'delayed fact' for text in texts):
                    outer.entered.set()
                    if not outer.release.wait(2):
                        raise TimeoutError('test embedding was not released')
                return [[0.1] * 16 for _ in texts]

            def embed_query_text(self, text):
                return [0.1] * 16

        self.env = patch.dict(os.environ, {'OPENAI_API_KEY': 'test-only'})
        self.env.start()
        self.store = Mem0BackendStore(Embedder(), memory_root=self.root)
        self.store._mem0.vector_store._bm25_encoder = False
        self.nlp = patch.multiple('mem0.memory.main',
            extract_entities=lambda _: [], lemmatize_for_bm25=lambda text: text)
        self.nlp.start()
        self.store.add_text('fixture', 'synthetic fact')

    def tearDown(self):
        self.release.set()
        self.nlp.stop()
        self.env.stop()
        with _MEM0_CLIENT_CACHE_LOCK:
            _MEM0_CLIENT_CACHE.pop(str((self.root / 'vectors').resolve()), None)
        self.store._mem0.vector_store.client.close()
        self.store._mem0.db.connection.close()
        self.store._mem0.llm.client.close()
        self.tmp.cleanup()

    def test_search_does_not_wait_for_background_embedding(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(self.store.add_text, 'fixture', 'delayed fact')
            try:
                self.assertTrue(self.entered.wait(1))
                hits = pool.submit(self.store.search, 'synthetic', user_id='fixture').result(timeout=1)
                self.assertEqual(hits[0].text, 'synthetic fact')
                self.assertFalse(writer.done())
            finally:
                self.release.set()
            self.assertTrue(writer.result(timeout=2))

    def test_same_directory_reuses_guarded_client_and_entity_store(self):
        other = Mem0BackendStore(self.store._embedder, memory_root=self.root)
        self.assertIs(other._mem0, self.store._mem0)
        with patch('mem0.vector_stores.qdrant.Qdrant._create_filter_indexes'):
            entity = self.store._mem0.entity_store
        self.assertIs(entity.client, self.store._mem0.vector_store.client)
        self.assertTrue(hasattr(entity.client.query_points, '__wrapped__'))

    def test_bm25_encoding_does_not_hold_the_native_client_lock(self):
        outer = self

        class Encoder:
            def embed(self, texts):
                if 'delayed keyword' in texts:
                    outer.entered.set()
                    if not outer.release.wait(2):
                        raise TimeoutError('test BM25 was not released')
                return [type('Sparse', (), {
                    'indices': np.array([1]), 'values': np.array([0.5])})() for _ in texts]

        vectors = self.store._mem0.vector_store
        vectors._bm25_encoder = Encoder()
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(vectors.insert, vectors=[[0.1] * 16],
                ids=[str(uuid.uuid4())], payloads=[{'data': 'delayed keyword', 'user_id': 'fixture'}])
            try:
                self.assertTrue(self.entered.wait(1))
                hits = pool.submit(self.store.search, 'synthetic', user_id='fixture').result(timeout=1)
                self.assertEqual(hits[0].text, 'synthetic fact')
                self.assertFalse(writer.done())
            finally:
                self.release.set()
            writer.result(timeout=2)


if __name__ == '__main__':
    unittest.main()
