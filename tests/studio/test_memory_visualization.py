"""Shared portrait labels resolve consistently without blocking display reads."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from studio.core.utils.perception.component import Perception
from studio.core.utils.visualization.component import Visualization
from studio.harness.speaking_style.policy import RB_HUMANIZE_PROMPT, RB_HUMANIZE_EXAMPLES
from studio.web.transport import build_app
from voicemem.rightbrain.traits_store import Evidence, Trait
from voicemem.utils.common.memory_changes import for_space
from studio.core.utils.visualization.owner_identity import current_owner_name


class DisplayAgent(Visualization, Perception):
    pass


class MemoryVisualizationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "multi_modal").mkdir()
        (self.root / "multi_modal" / "voiceprint_registry.json").write_text(json.dumps({
            "fixture-voice": {"role": "user", "name": "Fixture Owner"},
        }))
        self.trait = Trait(
            id="fixture-trait", slot="表达风格", claim="Prefers short replies",
            evidence=[Evidence(quote="Keep it brief.", emotion="平静", cause="A request for brevity")],
        )
        self.store = SimpleNamespace(all=Mock(return_value=[self.trait]), counts=Mock(return_value=(1, 1)))
        repo = self.repo = SimpleNamespace(
            _vector_store=SimpleNamespace(list_entries=Mock(return_value=[])),
            _cognitive_store=SimpleNamespace(memory_ids_for_slots=Mock(return_value=[])),
        )
        self.agent = DisplayAgent()
        self.agent.vm = SimpleNamespace(_o=SimpleNamespace(
            _user_id="fixture-user", _get_repo=lambda: repo, _memory_root=self.root,
            _right=SimpleNamespace(_traits=lambda: self.store),
        ))
        self.agent.ACTIVE_SPACE = "fixture-space"
        self.agent.SPACE_LANG = "zh"
        self.agent.RB_ENTITIES_PER_SLOT = 6
        self.agent.LB_ENTRIES_PER_SLOT = 7
        self.agent._LAST_HIT_IDS = set()
        self.agent.RB_HUMANIZE = True
        self.agent._OWNER_NAME_CACHE = {}
        self.agent.space_dir = lambda space: (self.root, "zh")
        self.agent._BAD_NAME_CHARS = '什谁哪啥吗呢么?？'
        self.agent._RB_HUMAN = {}
        self.agent._RB_HUMAN_PENDING = set()
        self.agent._RB_HUMAN_LOCK = threading.Lock()
        self.agent._RB_HUMANIZE_PROMPT = RB_HUMANIZE_PROMPT
        self.agent._RB_HUMANIZE_EXAMPLES = RB_HUMANIZE_EXAMPLES
        self.label = "Fixture Owner likes brief answers"

    @contextmanager
    def controlled_rewrite(self, content=None, error=None):
        started, release, finished = (threading.Event() for _ in range(3))
        original = self.agent._rb_humanize_now

        def create(**kwargs):
            started.set()
            if not release.wait(3):
                raise TimeoutError("Synthetic rewrite was not released")
            if error:
                raise error
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=self.label if content is None else content,
            ))])

        def run(*args):
            try:
                return original(*args)
            finally:
                finished.set()

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=create))))
        with patch("openai.OpenAI", return_value=client), \
                patch("voicemem.llm_config.resolve_model", return_value="fixture-model"), \
                patch.object(self.agent, "_rb_humanize_now", side_effect=run) as worker:
            try:
                yield SimpleNamespace(started=started, release=release, finished=finished,
                                      worker=worker, create=client.chat.completions.create)
            finally:
                release.set()
                if worker.called:
                    self.assertTrue(finished.wait(3), "Synthetic rewrite did not finish")

    def test_both_style_endpoints_share_pending_and_prepared_labels(self):
        app = build_app("llm_tts", None, None, snapshot=self.agent.memory_snapshot)
        client = TestClient(app)
        self.addCleanup(client.close)
        with self.controlled_rewrite() as rewrite:
            snapshots = []
            for style in ("digital", "technical"):
                self.assertEqual(client.get(f"/ui/{style}.html").status_code, 200)
                response = client.get("/api/memories")
                self.assertEqual(response.status_code, 200)
                snapshots.append(response.json())
            self.assertTrue(rewrite.started.wait(1))
            self.assertEqual(snapshots[0], snapshots[1])
            self.assertTrue(snapshots[0]["right"][0]["label_pending"])
            self.assertEqual(snapshots[0]["right"][0]["text"], self.trait.claim)

            rewrite.release.set()
            self.assertTrue(rewrite.finished.wait(1))
            ready = [client.get("/api/memories").json() for _ in range(2)]
            self.assertEqual(ready[0], ready[1])
            portrait = ready[0]["right"][0]
            self.assertFalse(portrait["label_pending"])
            self.assertEqual(portrait["text"], self.label)
            self.assertEqual(portrait["raw"], self.trait.claim)
            self.assertEqual(portrait["notes"][0]["text"], "Keep it brief.")
            rewrite.create.assert_called_once()
            self.assertIn("Fixture Owner", rewrite.create.call_args.kwargs["messages"][0]["content"])

    def test_concurrent_reads_share_one_nonblocking_rewrite(self):
        with self.controlled_rewrite() as rewrite, ThreadPoolExecutor(max_workers=8) as readers:
            results = [readers.submit(self.agent.memory_snapshot) for _ in range(8)]
            snapshots = [result.result(timeout=1) for result in results]
            self.assertTrue(all(s["right"][0]["label_pending"] for s in snapshots))
            self.assertTrue(rewrite.started.wait(1))
            rewrite.worker.assert_called_once()
            self.assertFalse(rewrite.finished.is_set())

    def test_failed_or_malformed_rewrite_resolves_to_shared_raw_fallback(self):
        for options in ({"error": RuntimeError("Synthetic provider failure")}, {"content": ""}):
            with self.subTest(options=options):
                self.agent._RB_HUMAN.clear()
                with self.controlled_rewrite(**options) as rewrite:
                    self.agent.memory_snapshot()
                    rewrite.release.set()
                    self.assertTrue(rewrite.finished.wait(1))
                    first, second = self.agent.memory_snapshot(), self.agent.memory_snapshot()
                    self.assertEqual(first, second)
                    self.assertEqual(first["right"][0]["text"], self.trait.claim)
                    self.assertFalse(first["right"][0]["label_pending"])
                    self.assertFalse(self.agent._RB_HUMAN_PENDING)
                    rewrite.create.assert_called_once()

    def test_background_completion_keeps_captured_owner_and_language(self):
        with self.controlled_rewrite() as rewrite:
            self.agent.memory_snapshot()
            self.assertTrue(rewrite.started.wait(1))
            self.agent.ACTIVE_SPACE = "another-fixture-space"
            self.agent.SPACE_LANG = "en"
            self.agent._OWNER_NAME_CACHE["another-fixture-space"] = "Another Owner"
            rewrite.release.set()
            self.assertTrue(rewrite.finished.wait(1))
            self.assertEqual(self.agent._RB_HUMAN, {
                ("zh", "Fixture Owner", self.trait.claim): self.label,
            })

    def test_request_language_does_not_change_space_or_original_memory(self):
        original = self.trait.claim = '喜欢简短回答'
        self.repo._vector_store.list_entries.return_value = [{
            'id': 'fixture-fact', 'text': '保留原始事实', 'date': '2026-01-01',
        }]
        app = build_app('llm_tts', None, None, snapshot=self.agent.memory_snapshot,
                        localized_snapshot=lambda language, include_all=False: self.agent.memory_snapshot(
                            language=language, include_all=include_all))
        client = TestClient(app)
        self.addCleanup(client.close)
        with self.controlled_rewrite() as rewrite:
            pending = client.get('/api/memories?lang=en').json()
            self.assertEqual(pending['language'], 'en')
            self.assertTrue(rewrite.started.wait(1))
            request = rewrite.create.call_args.kwargs['messages'][0]['content']
            self.assertIn('Write every display label in English', request)
            rewrite.release.set()
            self.assertTrue(rewrite.finished.wait(1))
            english = client.get('/api/memories?lang=en').json()
            self.assertEqual(english['right'][0]['text'], self.label)
            self.assertEqual(english['right'][0]['raw'], original)
            self.assertEqual(english['left'][0]['text'], '保留原始事实')
            full = client.get('/api/memories?all=true&lang=en').json()
            self.assertEqual(full['right'][0]['text'], self.label)
        with self.controlled_rewrite(content='喜欢简短回答的展示标签') as rewrite:
            chinese = client.get('/api/memories?lang=zh').json()
            self.assertEqual(chinese['right'][0]['text'], original)
            self.assertTrue(rewrite.started.wait(1))
            self.assertIn('每条展示标签用简洁自然的中文', rewrite.create.call_args.kwargs['messages'][0]['content'])
        self.assertEqual(self.agent.SPACE_LANG, 'zh')
        self.assertEqual(self.trait.claim, original)
        self.assertEqual(client.get('/api/memories?lang=invalid').status_code, 400)

    def test_language_parameter_keeps_legacy_snapshot_injection(self):
        read = Mock(return_value={'left': [], 'right': []})
        client = TestClient(build_app('llm_tts', None, None, snapshot=read))
        self.addCleanup(client.close)
        self.assertEqual(client.get('/api/memories?lang=en').status_code, 200)
        read.assert_called_once_with()

    def test_recall_keeps_name_rewriting_and_original_evidence(self):
        payload = {"emotion": "平静", "right_brain_hits": [
            {"claim": self.trait.claim, "content": f"{self.trait.claim}｜他说过：Keep it brief."},
        ]}
        self.agent._RB_HUMAN[("zh", "Fixture Owner", self.trait.claim)] = self.label
        with patch.object(self.agent, "_rb_humanize_now") as rewrite:
            result = self.agent.fill_tags(payload, "", acoustic=False)
        self.assertEqual(result["right_brain_hits"][0]["content"], f"{self.label}｜他说过：Keep it brief.")
        self.assertEqual(result["right_brain_hits"][0]["claim"], self.trait.claim)
        rewrite.assert_not_called()

    def test_disabled_rewriting_is_ready_without_starting_work(self):
        self.agent.RB_HUMANIZE = False
        with patch.object(self.agent, "_rb_humanize_now") as rewrite:
            portrait = self.agent.memory_snapshot()["right"][0]
        self.assertEqual(portrait["text"], self.trait.claim)
        self.assertFalse(portrait["label_pending"])
        rewrite.assert_not_called()

    def test_committed_rename_and_delete_invalidate_projection_without_changing_registry(self):
        registry = self.root / "multi_modal" / "voiceprint_registry.json"
        original = registry.read_bytes()
        self.assertEqual(self.agent.owner_name(), "Fixture Owner")
        self.repo._vector_store.list_entries.return_value = [
            {"id": "old", "text": "用户的名字是林舟。", "updated_at": "2026-01-01"},
            {"id": "new", "text": "用户的名字是陈河。", "updated_at": "2026-01-02"},
        ]
        for_space(self.root).publish(["new"])
        self.assertEqual(self.agent.owner_name(), "陈河")
        self.assertEqual(registry.read_bytes(), original)
        self.repo._vector_store.list_entries.return_value = []
        for_space(self.root).publish(["new"])
        self.assertEqual(self.agent.owner_name(), "Fixture Owner")

    def test_latest_fact_survives_graph_caps_and_full_view_keeps_older_facts(self):
        self.agent.RB_HUMANIZE = False
        rows = [{"id": f"fact-{index}", "text": f"Synthetic fact {index}", "date": "2026-01-01"}
                for index in range(80)]
        self.repo._vector_store.list_entries.return_value = rows
        self.assertEqual(len(self.agent.memory_snapshot()["left"]), 7)
        for_space(self.root).publish(["fact-79"])
        snapshot = self.agent.memory_snapshot(limit=3)
        self.assertEqual(snapshot["left"][0]["id"], "fact-79")
        self.assertEqual(snapshot["total"]["left"], 80)
        self.assertEqual(len(snapshot["left"]), 3)
        self.agent._LAST_HIT_IDS = {f"fact-{index}" for index in range(5)}
        crowded = {row["id"] for row in self.agent.memory_snapshot(limit=3)["left"]}
        self.assertTrue(self.agent._LAST_HIT_IDS.issubset(crowded))
        self.assertIn("fact-79", crowded)
        self.assertEqual(len(self.agent.memory_snapshot(include_all=True)["left"]), 80)
        self.repo._vector_store.list_entries.return_value = rows[:-1]
        for_space(self.root).publish(["fact-79"])
        self.assertNotIn("fact-79", {row["id"] for row in self.agent.memory_snapshot()["left"]})

    def test_full_view_does_not_generate_extra_portrait_labels(self):
        with patch.object(self.agent, "_rb_humanize_now") as rewrite:
            snapshot = self.agent.memory_snapshot(include_all=True)
        self.assertFalse(snapshot["right"][0]["label_pending"])
        rewrite.assert_not_called()

    def test_missing_optional_metadata_keeps_committed_facts_visible(self):
        self.agent.RB_HUMANIZE = False
        self.repo._vector_store.list_entries.return_value = [{"id": "fact", "text": "Committed fact"}]
        for store in (None, SimpleNamespace(memory_ids_for_slots=Mock(side_effect=RuntimeError("Synthetic metadata failure")))):
            with self.subTest(store=store):
                self.repo._cognitive_store = store
                snapshot = self.agent.memory_snapshot()
                self.assertEqual(snapshot["left"][0]["text"], "Committed fact")

    def test_unavailable_fact_store_keeps_voice_registration_fallback(self):
        self.repo._vector_store.list_entries.side_effect = RuntimeError("Synthetic read failure")
        self.assertEqual(self.agent.owner_name(), "Fixture Owner")

    def test_snapshot_keeps_captured_space_when_selection_changes_during_read(self):
        self.agent.RB_HUMANIZE = False
        def read(**kwargs):
            self.agent.vm = SimpleNamespace()
            self.agent.ACTIVE_SPACE = "other-space"
            self.agent.SPACE_LANG = "en"
            return []
        self.repo._vector_store.list_entries.side_effect = read
        snapshot = self.agent.memory_snapshot()
        self.assertEqual(snapshot["space"], "fixture-space")
        self.assertEqual(snapshot["right"][0]["raw"], self.trait.claim)


class OwnerProjectionTests(unittest.TestCase):
    def test_explicit_names_only_and_latest_change_wins(self):
        for text, expected in [
            ("用户的名字是林舟（Lin）。", "林舟"),
            ("用户改名为陈河。", "陈河"),
            ("User's name is Taylor.", "Taylor"),
            ("The user has changed their name to Morgan.", "Morgan"),
            ("用户的朋友叫林舟。", ""),
            ("用户以前叫林舟。", ""),
            ("User's name is Taylor?", ""),
            ("User's name is not Taylor.", ""),
        ]:
            with self.subTest(text=text):
                self.assertEqual(current_owner_name([{"text": text}]), expected)
        self.assertEqual(current_owner_name([
            {"text": "用户的名字是林舟。", "updated_at": "2026-01-01"},
            {"text": "用户的名字是陈河。", "updated_at": "2026-01-02"},
            {"text": "用户的名字是他人。", "updated_at": "2026-01-03", "role": "assistant"},
        ]), "陈河")


if __name__ == "__main__":
    unittest.main()
