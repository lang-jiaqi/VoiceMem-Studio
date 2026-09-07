"""Retrieval-scoped encoding reuse without result caching or model changes."""
import asyncio
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicemem.leftbrain.query_embedding import encode_query, query_embedding_scope


class Model:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kw):
        self.calls.append((list(texts), kw))
        return np.array([[1., 2., 3.] for _ in texts])


class QueryEmbeddingTests(unittest.TestCase):
    def test_identical_query_reused_but_returned_vectors_are_independent(self):
        model = Model()
        with query_embedding_scope():
            first = encode_query(model, "query: hello")
            first[0] = -1
            second = encode_query(model, "query: hello")
            second[1] = -2
            np.testing.assert_equal(encode_query(model, "query: hello"), [1., 2., 3.])
        self.assertEqual(len(model.calls), 1)

    def test_models_prefixes_changed_queries_and_new_requests_never_collide(self):
        one, two = Model(), Model()
        with query_embedding_scope():
            for text in ("query: hello", "passage: hello", "query: tomorrow"):
                encode_query(one, text)
            encode_query(two, "query: hello")
        with query_embedding_scope():
            encode_query(one, "query: hello")
        encode_query(one, "query: hello")
        self.assertEqual(len(one.calls), 5)
        self.assertEqual(len(two.calls), 1)

    def test_exception_restores_scope(self):
        model = Model()
        with query_embedding_scope():
            encode_query(model, "hello")
            with self.assertRaises(ValueError):
                with query_embedding_scope():
                    encode_query(model, "hello")
                    raise ValueError()
            encode_query(model, "hello")
        self.assertEqual(len(model.calls), 2)
        encode_query(model, "hello")
        self.assertEqual(len(model.calls), 3)

    def test_actual_classifier_and_embedder_share_exact_encoding(self):
        from voicemem.leftbrain.local_embedder import LocalEmbedder, REGISTRY
        from voicemem.leftbrain.cognitive_graph.local_query_classifier import LocalQueryClassifier
        model = Model()
        classifier = LocalQueryClassifier(model=model, language="zh")
        classifier._spec = REGISTRY["e5"]
        classifier._slot_names = ["one", "two", "three"]
        classifier._slot_embs = np.eye(3)
        embedder = LocalEmbedder.__new__(LocalEmbedder)
        embedder.model, embedder._path = REGISTRY["e5"], "test"
        with patch("voicemem.leftbrain.local_embedder.shared_model", return_value=model):
            with query_embedding_scope():
                result = classifier.classify("你好")
                vector = embedder.embed_query_text("你好")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result.slots, ["three", "two"])
        self.assertEqual(vector, [1., 2., 3.])


class StreamSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_speculation_scopes_reuse_and_keeps_results(self):
        from voicemem.stream import VoiceStream
        model = Model()
        result = types.SimpleNamespace(hits=[], timing={})
        def classify(text):
            encode_query(model, text)
            return types.SimpleNamespace(slots=["work"], entities=[])
        def search(text, **kw):
            encode_query(model, text)
            self.assertEqual(kw["slots"], ["work"])
            return result
        stream = VoiceStream(types.SimpleNamespace(classify=classify, search=search),
                             gate=lambda _: "deep")
        turn = await stream._speculate("你好")
        self.assertIs(turn.result, result)
        self.assertEqual(len(model.calls), 1)
        await stream._speculate("你好")
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
