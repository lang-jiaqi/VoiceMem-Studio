"""Repository ownership regressions with synthetic vectors and temporary SQLite files."""
import importlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from voicemem.leftbrain.cognitive_graph.annotator import NullAnnotator
from voicemem.leftbrain.cognitive_graph.slot_v2 import (
    ALL_SLOT_V2_VALUES, SLOT_V2_DESCRIPTIONS,
)
from voicemem.leftbrain.cognitive_graph.store import CognitiveGraphStore
from voicemem.leftbrain.cognitive_graph.store_v2 import CognitiveGraphStoreV2
from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
from voicemem.leftbrain.local_memory_store import MemorySearchHit
from voicemem.leftbrain.memory_repository import (
    LeftBrainMemoryRepository, LeftBrainMemoryRepositoryConfig,
)
from voicemem.leftbrain.memory_repository_v2 import LeftBrainMemoryRepositoryV2


class FixtureEmbedder:
    def embed_texts(self, texts):
        vectors = []
        for text in texts:
            slot = next((s for s, description in SLOT_V2_DESCRIPTIONS.items()
                         if text == description), None)
            slot = slot or ('work' if 'work' in text or '工作' in text else 'knowledge')
            vectors.append([float(s == slot) for s in ALL_SLOT_V2_VALUES])
        return vectors


class FixtureVectors:
    def __init__(self):
        self.rows = {}
        self.calls = []
        self.next_id = 0

    def add_records_with_ids(self, user_id, items):
        ids = []
        for _, text, attributed_to, metadata in items:
            self.next_id += 1
            mid = f'backend-{self.next_id}'
            self.rows[mid] = dict(text=text, user_id=user_id,
                                  attributed_to=attributed_to, metadata=dict(metadata))
            ids.append(mid)
        return ids

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        rows = ((mid, row) for mid, row in self.rows.items()
                if row['user_id'] == kwargs['user_id'])
        return [MemorySearchHit(mid, row['text'], 1., row['attributed_to'], row['metadata'])
                for mid, row in rows][:kwargs['top_k']]

    def update_memory(self, memory_id, new_text, **kwargs):
        if memory_id not in self.rows:
            return False
        self.rows[memory_id]['text'] = new_text
        return True

    def delete_memory(self, memory_id):
        return self.rows.pop(memory_id, None) is not None


class RepositoryOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='voicemem-structure-')
        root = Path(self.tmp.name)
        self.config = LeftBrainMemoryRepositoryConfig(
            json_path=root / 'memories.json', db_path=root / 'vectors.sqlite',
            cognitive_db_path=root / 'cognitive.sqlite', enable_cognitive_graph=True,
        )
        self.vectors = FixtureVectors()
        self.embedder = FixtureEmbedder()
        slots = importlib.import_module('voicemem.leftbrain.memory_repository_v2')
        self.cache = patch.object(slots, '_SLOT_EMBED_CACHE', {})
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.addCleanup(self.tmp.cleanup)

    def repository(self, cls=LeftBrainMemoryRepositoryV2):
        return cls(self.embedder, config=self.config,
                   cognitive_annotator=NullAnnotator(), vector_store=self.vectors)

    def fact(self, text):
        return ExtractedAdditiveMemory('extractor-id', text, 'user')

    def test_repository_types_keep_their_original_implementation_modules(self):
        self.assertEqual(LeftBrainMemoryRepository.__module__,
                         'voicemem.leftbrain.memory_repository')
        self.assertEqual(LeftBrainMemoryRepositoryV2.__module__,
                         'voicemem.leftbrain.memory_repository_v2')
        from voicemem.leftbrain import LeftBrainMemoryRepositoryConfig as exported
        self.assertIs(exported, LeftBrainMemoryRepositoryConfig)
        from voicemem.leftbrain.memory_repository import create_openai_memory_repository
        self.assertTrue(callable(create_openai_memory_repository))
        self.assertTrue(issubclass(LeftBrainMemoryRepositoryV2, LeftBrainMemoryRepository))

    def test_slot_repository_initializes_each_schema_once_and_keeps_embedder(self):
        calls = []
        def recording(method, label):
            def run(store):
                calls.append(label)
                return method(store)
            return run
        with patch.object(CognitiveGraphStore, '_ensure_schema',
                          recording(CognitiveGraphStore._ensure_schema, 'base')), \
             patch.object(CognitiveGraphStore, '_migrate_schema',
                          recording(CognitiveGraphStore._migrate_schema, 'migration')), \
             patch.object(CognitiveGraphStoreV2, '_ensure_tags_schema',
                          recording(CognitiveGraphStoreV2._ensure_tags_schema, 'tags')):
            repo = self.repository()
        self.assertEqual(calls, ['base', 'migration', 'tags'])
        self.assertIs(repo.cognitive_store._embedder, self.embedder)
        self.assertIs(repo.vector_store, self.vectors)
        self.assertIs(type(repo.cognitive_store), CognitiveGraphStoreV2)

    def test_base_repository_keeps_base_store_and_graph_can_remain_disabled(self):
        self.assertIs(type(self.repository(LeftBrainMemoryRepository).cognitive_store),
                      CognitiveGraphStore)
        self.config.enable_cognitive_graph = False
        repo = self.repository()
        self.assertIsNone(repo.cognitive_store)
        self.assertEqual(repo.append_extracted([self.fact('knowledge')], user_id='a'),
                         ['backend-1'])

    def test_bilingual_writes_keep_backend_ids_mirror_graph_tags_and_user_isolation(self):
        repo = self.repository()
        ids = repo.append_extracted(
            [self.fact(''), self.fact('I work remotely.'), self.fact('我喜欢这份工作。')],
            user_id='a', extra_metadata={'session_id': 'fixture'},
        )
        self.assertEqual(ids, ['backend-1', 'backend-2'])
        mirror = repo.load_json_store()['results']
        self.assertEqual([row['id'] for row in mirror], ids)
        self.assertEqual([row['memory'] for row in mirror],
                         ['I work remotely.', '我喜欢这份工作。'])
        for mid in ids:
            self.assertEqual(repo.cognitive_store.get_memory_record(mid).content,
                             self.vectors.rows[mid]['text'])
        self.assertEqual(set(repo.cognitive_store.memory_ids_for_slots_v2('a', ['work'])),
                         set(ids))
        self.assertEqual(repo.cognitive_store.memory_ids_for_slots_v2('b', ['work']), [])
        self.assertEqual(repo.existing_for_extractor(user_id='b'), [])

    def test_update_delete_and_missing_ids_keep_existing_contracts(self):
        repo = self.repository()
        mid = repo.append_extracted([self.fact('Old knowledge')], user_id='a')[0]
        with patch.object(repo._cognitive_annotator, 'annotate',
                          wraps=repo._cognitive_annotator.annotate) as annotate:
            self.assertTrue(repo.update_memory(mid, 'I work remotely.', user_id='a',
                                              observed_at='2026-01-01'))
        annotate.assert_called_once_with(['I work remotely.'])
        mirror = repo.load_json_store()['results'][0]
        self.assertEqual(mirror['memory'], 'I work remotely.')
        self.assertEqual(mirror['created_at'], '2026-01-01')
        self.assertEqual(self.vectors.rows[mid]['text'], 'I work remotely.')
        self.assertEqual(repo.cognitive_store.memory_ids_for_slots_v2('a', ['work']), [mid])
        self.assertFalse(repo.update_memory('missing', 'text', user_id='a'))
        self.assertTrue(repo.delete_memory(mid))
        self.assertEqual(repo.load_json_store()['results'], [])
        self.assertFalse(repo.delete_memory(mid))

    def test_search_delegates_without_changing_query_or_provider_options(self):
        repo = self.repository()
        repo.search('原始 query', user_id='a', top_k=7, threshold=.42,
                    include_assistant=True)
        self.assertEqual(self.vectors.calls, [('原始 query', dict(
            user_id='a', top_k=7, threshold=.42, include_assistant=True))])

    def test_retired_repository_entry_points_fail_before_search_or_background_work(self):
        from typing import get_type_hints
        for cls in (LeftBrainMemoryRepository, LeftBrainMemoryRepositoryV2):
            repo = self.repository(cls)
            with patch.object(repo, 'search', side_effect=AssertionError('unexpected query')), \
                 patch('concurrent.futures.ThreadPoolExecutor',
                       side_effect=AssertionError('unexpected executor')):
                for name in ('search_with_graph', 'search_combined'):
                    with self.subTest(repository=cls.__name__, method=name):
                        method = getattr(repo, name)
                        get_type_hints(method)
                        with self.assertRaisesRegex(NotImplementedError, r'VoiceMem\.search\(\)'):
                            method('fixture', user_id='a')
            self.assertEqual(self.vectors.calls, [])

    def test_optional_fusion_uses_existing_plain_search_for_retired_graph_api(self):
        from voicemem.utils.fusion.config import FusionConfig
        from voicemem.utils.fusion.left_channel import search_left_for_reply
        repo = self.repository()
        mid = repo.append_extracted([self.fact('knowledge')], user_id='a')[0]
        hits, appendix = search_left_for_reply(
            repo, asr_text='fixture', user_id='a',
            config=FusionConfig(mem0_top_k=3, mem0_threshold=.25))
        self.assertEqual([hit.memory_id for hit in hits], [mid])
        self.assertEqual(appendix, '')
        self.assertEqual(self.vectors.calls, [('fixture', dict(
            user_id='a', top_k=3, threshold=.25, include_assistant=False))])

    def test_optional_fusion_preserves_custom_graph_results_and_real_errors(self):
        from types import SimpleNamespace
        from voicemem.utils.fusion.left_channel import search_left_for_reply
        memory = MemorySearchHit('fixture-id', 'knowledge', .8, 'user', {})
        provider = SimpleNamespace(
            search=Mock(side_effect=AssertionError('unexpected fallback')),
            search_with_graph=Mock(return_value=[SimpleNamespace(memory=memory, graph=None)]))
        hits, _ = search_left_for_reply(provider, asr_text='fixture', user_id='a')
        self.assertEqual([hit.memory_id for hit in hits], ['fixture-id'])
        provider.search.assert_not_called()
        provider.search_with_graph.side_effect = ValueError('synthetic graph failure')
        with self.assertRaisesRegex(ValueError, 'synthetic graph failure'):
            search_left_for_reply(provider, asr_text='fixture', user_id='a')
        provider.search.assert_not_called()

    def test_optional_fusion_handles_unsupported_graph_without_threshold_parameter(self):
        from types import SimpleNamespace
        from voicemem.utils.fusion.left_channel import search_left_for_reply
        def graph(query, *, user_id, top_k):
            raise NotImplementedError('synthetic unsupported capability')
        provider = SimpleNamespace(search_with_graph=graph, search=Mock(return_value=[]))
        self.assertEqual(search_left_for_reply(provider, asr_text='fixture', user_id='a'),
                         ([], ''))
        provider.search.assert_called_once()

    def test_existing_graph_upgrades_and_reopens_without_rewriting_memories(self):
        base = self.repository(LeftBrainMemoryRepository)
        mid = base.append_extracted([self.fact('已有知识')], user_id='a')[0]
        before = base.load_json_store()
        record = base.cognitive_store.get_memory_record(mid)
        slots = self.repository()
        self.assertEqual(slots.load_json_store(), before)
        self.assertEqual(slots.cognitive_store.get_memory_record(mid), record)
        slots.cognitive_store.upsert_slot_summary('a', 'knowledge', 'fixture summary', 1)
        reopened = self.repository()
        self.assertEqual(reopened.cognitive_store.get_slot_summaries('a', ['knowledge']),
                         {'knowledge': 'fixture summary'})
        self.assertEqual(reopened.load_json_store(), before)


class SharedTraceOwnershipTests(unittest.TestCase):
    def test_api_reply_and_studio_use_the_shared_request_journal(self):
        shared = importlib.import_module('voicemem.utils.common.prompt_trace')
        from voicemem.reply import record_request
        self.assertIs(record_request, shared.record_request)
        root = Path(__file__).resolve().parents[2]
        for name in ('studio/core/core.py', 'studio/core/utils/reply/component.py',
                     'studio/core/utils/tts/component.py', 'studio/core/utils/tts/cuda.py'):
            self.assertIn('from voicemem.utils.common.prompt_trace import',
                          (root / name).read_text())

    def test_core_imports_and_api_reply_do_not_import_studio_or_models(self):
        script = '''
import json, sys
import voicemem
from voicemem import VoiceMem
import voicemem.reply
print(json.dumps([m for m in sys.modules if m == 'studio' or m.startswith('studio.')
                 or m in ('torch', 'mlx', 'sentence_transformers')]))
'''
        run = subprocess.run([sys.executable, '-c', script],
                             cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(run.stdout), [])


class PublicModuleOwnershipTests(unittest.TestCase):
    def test_package_root_contains_implementations_without_module_redirects(self):
        root = Path(__file__).resolve().parents[2] / 'voicemem'
        expected = {'__init__', 'audio_timing', 'config', 'core', 'gate', 'lang',
                    'llm_config', 'memory_api', 'orchestrator', 'reply',
                    'startup_check', 'stream', 'tts'}
        self.assertEqual({p.stem for p in root.glob('*.py')}, expected)
        for path in root.rglob('*.py'):
            self.assertNotIn('sys.modules[__name__]', path.read_text(), str(path))

    def test_public_facade_reexports_the_same_capabilities_and_result_types(self):
        import voicemem
        from voicemem.core import SearchResult as facade_result, Utils as facade_utils
        from voicemem.orchestrator import Utils, SearchResult
        from voicemem.stream import StreamState, Turn
        self.assertIs(voicemem.Utils, facade_utils)
        self.assertIs(facade_utils, Utils)
        self.assertIs(voicemem.SearchResult, facade_result)
        self.assertIs(facade_result, SearchResult)
        self.assertIs(voicemem.Turn, Turn)
        self.assertIs(voicemem.StreamState, StreamState)

    def test_old_serialized_turn_and_result_paths_still_load(self):
        from voicemem.stream import Turn, empty_result
        from voicemem.orchestrator import SearchResult
        turn = Turn('fixture', empty_result(), raw_text='partial', route='shallow')
        self.assertEqual(Turn.__module__, 'voicemem.stream')
        self.assertEqual(SearchResult.__module__, 'voicemem.orchestrator')
        serialized = pickle.dumps(turn)
        restored = pickle.loads(serialized)
        self.assertEqual(restored, turn)
        self.assertIs(type(restored), Turn)
        self.assertIs(type(restored.result), SearchResult)

    def test_state_properties_reuse_existing_results_and_lazy_perception(self):
        from types import SimpleNamespace
        import voicemem.stream as owner
        result = SimpleNamespace(
            hits=[], rb_hits=[], classification=SimpleNamespace(
                slots=['work'], entities=['fixture']), scene_directive='', rb_directive='')
        snapshot = owner.StreamState('<silence>', 'fixture', result, None,
                                    _vm=object(), _pcm=[0.])
        perceived = SimpleNamespace(emotion='calm', person_id='speaker-fixture')
        with patch.object(owner, '_perceive', return_value=perceived) as perceive:
            self.assertEqual(snapshot.entity, ['fixture'])
            self.assertEqual(snapshot.schema, ['work'])
            perceive.assert_not_called()
            self.assertEqual(snapshot.emotion, 'calm')
            self.assertEqual(snapshot.speaker_id, 'speaker-fixture')
            perceive.assert_called_once()


if __name__ == '__main__':
    unittest.main()
