"""Coordinate factual memory, affective memory and audio perception.

One Orchestrator owns the Space identity, component cache, lock and recent
exchanges. Components implement their own domain logic. This module preserves
cross-component search, ingestion, completion callbacks and session flushing."""

from __future__ import annotations
from voicemem.lang import scoped_operation, contextualize

from voicemem.utils.common import space as _space

import functools
import inspect
import os
import time
import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
from voicemem.leftbrain.local_memory_store import MemorySearchHit
# LeftBrain owns candidate selection and ranking; this module coordinates components.
from voicemem.leftbrain.brain import LeftBrain, _search_mode
from voicemem.utils.audio.perceiver import AudioPerception, AudioPerceiver
# RightBrain owns affective retrieval and its prompt-ready directives.
from voicemem.rightbrain.brain import (
    RightBrain,
    RightBrainHit,
    _is_en_text,
    _rb_blended_priority,
    _rb_ctx_to_hits,
    _rb_lang,
    _rb_mem_date,
    _rb_trait_hits,
    _render_rb_directive,
)
from voicemem.utils.defaults import default_utils
from voicemem.llm_config import resolve_api_key, resolve_base_url, resolve_model

# Consolidate after this many new factual or affective memories. Session boundaries also flush pending work.
SHORT_TERM_MIN_MEMORIES = int(os.environ.get("VOICEMEM_ATTRIBUTION_MIN_MEMORIES", "20"))


# Time window in which a sound-only turn may inherit a preceding music intent.
EXPECT_MUSIC_S = float(os.environ.get("VOICEMEM_EXPECT_MUSIC_S", "120"))


_NEED = {
    "left_brain_single": ["embedding", "slots", "entity", "memory_engine"],
    "text_mode":         ["embedding", "slots", "entity", "emotion", "memory_engine"],
    "multi_modal":       ["embedding", "slots", "entity", "emotion", "voiceprint", "asr", "memory_engine"],
}

_ALIASES = {"schema": "slots", "embedder": "embedding",
            "vector_store": "memory_engine", "classifier": "slots"}

def _canon(overrides: dict) -> dict:
    """Normalize legacy capability names; explicit canonical keys take precedence."""
    out = {}
    for k, v in overrides.items():
        out.setdefault(_ALIASES.get(k, k), v)
    for k, v in overrides.items():
        if k not in _ALIASES:
            out[k] = v
    return out

class Utils:
    """Per-instance capability factories and objects, resolved lazily and cached once."""
    def __init__(self, mode, base_url, memory_root, overrides):
        self._factory = {**default_utils(base_url, memory_root), **_canon(overrides)}
        self.need = _NEED[mode]
        self._cache = {}
    def get(self, name):
        """Return the cached capability, constructing it if needed.

        Functions, methods, classes and partials are factories. Other objects are used
        as provided, even when they implement __call__. Construction errors propagate."""
        if name not in self._cache:
            f = self._factory[name]
            self._cache[name] = f() if (inspect.isfunction(f) or inspect.ismethod(f)
                                        or inspect.isclass(f)
                                        or isinstance(f, functools.partial)) else f
        return self._cache[name]


@dataclass
class SearchResult:
    """Combined retrieval result with factual hits, affective hits and stage timings."""
    hits: list[MemorySearchHit]
    classification: QueryClassification
    related_summaries: dict[str, str]   # {slot: summary_text}
    slot_mem_ids: set[str]  # Initial slot candidates.
    final_candidate_ids: set[str]  # Candidates after entity selection.
    search_mode: str = "fallback"
    rb_directive: str = ""  # Prompt-ready affective guidance.
    rb_hits: list[RightBrainHit] = field(default_factory=list)  # Structured affective hits.
    scene_directive: str = ""  # Scene-specific response guidance.
    current_scene: str = ""  # Current scene tag.
    timing: dict = None  # Stage durations in milliseconds.


    @property
    def result_leftbrain(self) -> list[str]:
        """Return factual text in the existing hit order."""
        return [h.text for h in self.hits]

    @property
    def result_rightbrain(self) -> list[str]:
        """Return affective context in the existing hit order."""
        return [h.content for h in self.rb_hits]


class Orchestrator:
    """Memory runtime composed from LeftBrain, RightBrain and AudioPerceiver.

    api_key sets the existing process environment key. mode selects capability
    requirements and default feature flags; explicit flags override those defaults.
    memory_root overrides Space directory resolution. user_id identifies the owner.
    base_url selects the OpenAI-compatible endpoint. Legacy embedder, vector_store
    and classifier arguments alias embedding, memory_engine and slots. Capability
    overrides accept lazy factories or existing objects."""

    def __init__(
        self,
        api_key: str | None = None,
        mode: str = "text_mode",
        memory_root: Path | str | None = None,
        space: str | None = None,
        user_id: str = "voice_user",
        base_url: str | None = None,
        enable_scene: bool | None = None,
        enable_music: bool | None = None,
        enable_abnormal_sound: bool | None = None,
        enable_voiceprint: bool | None = None,
        enable_emotion: bool | None = None,
        embedder: Any = None,
        vector_store: Any = None,
        classifier: Any = None,
        **util_overrides,
    ) -> None:
        if mode not in _NEED:
            raise ValueError("mode 必须是 " + " / ".join(_NEED))
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        self.mode = mode
        # Normalize legacy capability names before constructing the lazy capability table.
        overrides = _canon({**util_overrides, "embedder": embedder,
                            "vector_store": vector_store, "classifier": classifier})
        overrides = {k: v for k, v in overrides.items() if v is not None}
        # Resolve the Space before creating factories: model selection may read its language.
        # Runtime data belongs to the selected Space, independent of the package installation.
        from voicemem.utils.common.space import MemorySpace
        if memory_root or os.environ.get("VOICEMEM_MEMORY_ROOT"):
            self._memory_root = Path(memory_root or os.environ["VOICEMEM_MEMORY_ROOT"])
            self._memory_root.mkdir(parents=True, exist_ok=True)
            self._space = None
        else:
            self._space = MemorySpace(space)
            self._memory_root = self._space.dir
        self.utils = Utils(mode, base_url, self._memory_root, overrides)

        audio = mode == "multi_modal"
        pick = lambda n: self.utils.get(n) if n in overrides else None

        # Explicit feature flags override the defaults implied by the memory mode.
        if enable_scene is None:          enable_scene = audio
        if enable_music is None:          enable_music = audio
        if enable_abnormal_sound is None: enable_abnormal_sound = audio
        if enable_voiceprint is None:     enable_voiceprint = audio
        if enable_emotion is None:        enable_emotion = mode != "left_brain_single"
        embedder     = pick("embedding")
        vector_store = pick("memory_engine")
        classifier   = pick("slots")

        self._vector_store = vector_store  # Injected backend; None selects the built-in memory engine.
        # SQLite metadata stores share the Space database; tables have distinct names.
        self._db_path = _space.db(self._memory_root)
        self._multi_modal = _space.mm(self._memory_root)
        _space.describe(self._memory_root, user_id=user_id, mode=mode)
        self._multi_modal.mkdir(parents=True, exist_ok=True)
        self._cognitive_db = self._db_path
        self._user_id = user_id
        self._base_url = resolve_base_url(base_url)
        # Official/default is OpenAI embeddings (OpenAILocalEmbedder, built
        # lazily in _get_repo() below); pass a different TextEmbedder-
        # conforming object here to use something else for the left-brain
        # store's raw-fact embedding (used for both ingest and Rank()'s
        # search-time ranking). openai_voice_demo uses this to swap in a
        # local model for speed -- see that demo's local_embedder.py.
        self._embedder = embedder
        # The injected classifier implements classify(); child-slot descent is optional.
        self._classifier = classifier

        self._enable_scene = enable_scene
        self._enable_music = enable_music
        self._enable_abnormal_sound = enable_abnormal_sound
        self._enable_voiceprint = enable_voiceprint
        self._enable_emotion = enable_emotion

        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._ingest_count = 0

        # Keep two exchanges for factual disambiguation and reaction attribution.
        # Readers take a snapshot because reply registration and retrieval use different threads.
        self._exchanges: deque[tuple[str, str]] = deque(maxlen=2)

        # Construct LeftBrain first. Audio and affective dependencies resolve its repository lazily.
        # Components share this runtime's existing cache and lock.
        self._left = LeftBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embedder=self._embedder,
            vector_store=self._vector_store,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            classifier=self._classifier,
            tracker=self._get_session_tracker,
            cache=self._cache,
            lock=self._lock,
        )

        # AudioPerceiver owns acoustic models and speaker bindings.
        # Inject access to memory stores instead of constructing another repository.
        self._audio = AudioPerceiver(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            enable_scene=self._enable_scene,
            enable_music=self._enable_music,
            enable_abnormal_sound=self._enable_abnormal_sound,
            enable_voiceprint=self._enable_voiceprint,
            enable_emotion=self._enable_emotion,
            repo=self._get_repo,
            extractor=self._get_extractor,
            registry=self._get_registry,
            tag=self._tag_memories,
            extract_and_append=self._extract_and_append,
            rank=self.Rank,
            ingest_env=lambda: self.IngestEnv,
            cache=self._cache,
            lock=self._lock,
        )

        # RightBrain owns affective stores. Cross-component callbacks resolve through this runtime.
        # Deferred callbacks preserve instance overrides and shared capability caches.
        self._right = RightBrain(
            memory_root=self._memory_root,
            user_id=self._user_id,
            base_url=self._base_url,
            cognitive_db=self._cognitive_db,
            embed=self._embed_text,
            llm_json=self._llm_json,
            llm_text=self._llm_text,
            tracker=self._get_session_tracker,
            repo=self._get_repo,
            generate_inner_os=lambda text, emotion, entities, agent_reply="": (
                self._generate_inner_os(text, emotion, entities, agent_reply)),
            extract_rb_traits=lambda text, emotion: self._extract_rb_traits(text, emotion),
            cache=self._cache,
            lock=self._lock,
        )

        self.left_brain = self._left
        self.right_brain = self._right

    # Lazy component access and compatibility delegates.

    def _get_repo(self):
        return self._left._get_repo()

    def _get_rb_repo(self):
        return self._right._rb_repo()

    def _get_extractor(self):
        return self._left._get_extractor()

    def _get_registry(self):
        with self._lock:
            if "registry" not in self._cache:
                from voicemem.utils.common.voice_input import VoiceprintRegistry
                self._cache["registry"] = VoiceprintRegistry(
                    _space.mm(self._memory_root, "voiceprint_registry.json"),
                    entity_resolver=self._person_entity_id,
                )
        return self._cache["registry"]

    def _person_entity_id(self, name: str) -> str:
        """Read a person entity ID by name; return an empty string without creating an entity."""
        try:
            from voicemem.leftbrain.cognitive_graph.store import normalize_name
            store = self._get_repo()._cognitive_store
            for e in store.find_entities(self._user_id, name_norm=normalize_name(name)):
                # Use the enum value as the persistent type key, rather than its display representation.
                if getattr(e.entity_type, "value", e.entity_type) in ("person", "user"):
                    return e.id
        except Exception:
            pass
        return ""

    # Audio capability access.

    def _get_env_detector(self):
        return self._audio._env_detector()

    def _clap_memory_enabled(self) -> bool:
        # AST always supplies the immediate hint. Once a CLAP checkpoint is
        # configured, the 4s-segmented CLAP pass takes over the background-sound
        # description memory write; set VOICEMEM_ENVIRONMENT_MEMORY_BACKEND=ast
        # to opt back out.
        return (
            os.environ.get("VOICEMEM_ENVIRONMENT_MEMORY_BACKEND", "clap").lower() == "clap"
            and bool(os.environ.get("VOICEMEM_CLAP_CHECKPOINT"))
        )

    def _get_clap_env_detector(self):
        return self._audio._clap_env_detector()

    def _finish_clap_environment(self, *a, **k) -> None:
        return self._audio._finish_clap_environment(*a, **k)

    def _get_trigger_store(self):
        return self._audio._trigger_store()

    def _get_audio_archive(self):
        return self._audio._audio_archive()

    def _get_speaker_encoder(self):
        return self._audio._speaker_encoder()

    def _get_vp_store(self):
        return self._audio._vp_store()

    def _get_emotion_detector(self):
        return self._audio._emotion_detector()

    def _get_music_store(self):
        return self._audio._music_store()

    def _get_routine_store(self):
        return self._audio._routine_store()

    def _get_place_store(self):
        return self._audio._place_store()

    # Speaker bindings remain owned by AudioPerceiver; these properties expose the same dictionaries.
    @property
    def _session_person_pin(self) -> dict[str, str]:
        return self._audio._session_person_pin

    @property
    def _person_origin_session(self) -> dict[str, str]:
        return self._audio._person_origin_session

    def _claimed_by_other_identity(self, *a, **k) -> bool:
        return self._audio._claimed_by_other_identity(*a, **k)

    def _reconcile_speaker_candidates(self, *a, **k) -> tuple[str, str]:
        return self._audio._reconcile_speaker_candidates(*a, **k)

    # Scene reminders and archived audio.

    def CreateSceneTrigger(self, *a, **k) -> dict:
        return self._audio.CreateSceneTrigger(*a, **k)

    def GetOriginalAudio(self, *a, **k) -> dict:
        return self._audio.GetOriginalAudio(*a, **k)

    def TryPlayback(self, *a, **k) -> dict | None:
        return self._audio.TryPlayback(*a, **k)

    # Dynamic slot access.

    def _get_dynamic_slot_store(self):
        return self._left._get_dynamic_slot_store()

    def _get_dynamic_slots(self) -> list[tuple[str, str]]:
        """Return the owner's dynamic slot names and descriptions from LeftBrain."""
        return self._left._get_dynamic_slots()

    # Entity graphs and session attribution.

    def _get_graph_entity_store(self):
        return self._left._get_graph_entity_store()

    def _get_rb_graph_store(self):
        return self._right._rb_graph_store()

    def _get_session_tracker(self):
        with self._lock:
            if "session_tracker" not in self._cache:
                from voicemem.utils.common.session_tracker import SessionTracker
                self._cache["session_tracker"] = SessionTracker(
                    _space.db(self._memory_root)
                )
        return self._cache["session_tracker"]

    def _get_subgraph_manager(self):
        return self._left._get_subgraph_manager()

    def _get_attribution_manager(self):
        return self._right._attribution_manager()


    def _extract_rb_traits(self, text: str, emotion: str) -> list[tuple[str, str]]:
        """Extract subjective trait labels in the operation language.

        Reuse merged-extraction results when available; otherwise use the existing LLM
        request. Canonical slot names and language filtering remain unchanged."""
        import json as _json

        from voicemem.lang import is_zh as _is_zh, label_rule as _label_rule

        def _looks_cjk(t: str) -> bool:
            return any("\u4e00" <= ch <= "\u9fff" for ch in (t or ""))
        def _keep(items):
            """Discard labels that do not match the operation's Chinese/English script rule."""
            want_cjk = _is_zh()
            out = []
            for slot, label in items:
                if _looks_cjk(label) != want_cjk:
                    print(f"[RBTrait] 语言不符，丢弃：{slot} ← {label}", flush=True)
                    continue
                out.append((slot, label))
            return out

        from voicemem.leftbrain import merged_extraction
        if merged_extraction.enabled():
            cached = merged_extraction.take_traits(text)
            if cached is not None:
                valid = {"喜好与厌恶", "表达风格", "思维模式", "应对方式", "情绪"}
                return _keep([(s, l) for s, l in cached if s in valid and l])

        # Choose examples using the operation language. Trait slot names remain canonical internal keys.
        if _is_zh():
            prompt = (
                f"用户说了这句话（当前情绪：{emotion or '未知'}）：\n「{text[:300]}」\n\n"
                "判断这句话有没有透露出以下几类主观信息，每类最多提炼一条简短标签"
                "（5-15 字）：\n"
                "- 喜好与厌恶：本能的喜欢/讨厌/偏好\n"
                "- 表达风格：说话/沟通方式和习惯\n"
                "- 思维模式：思考、判断、决策的习惯\n"
                "- 应对方式：面对压力/负面情绪时怎么自我调节\n"
                "- 情绪：什么情况下会有什么情绪。**必须写成一个规律，不是一个情绪词**：\n"
                "  「评审前会紧张」「被打断就烦」「一个人待着会踏实」，不要写「焦虑」「开心」。\n"
                "  这句话会成为脑图上一个节点的标题，光一个情绪词看不出是什么事。\n\n"
                "没有清晰体现的类别就不要输出。\n"
                "**标签的写法**：写成一句短短的规律，5-15 字，不要主语、不要句号：\n"
                "「讨厌被打断」「压力大时想被安抚」「先要结论」。\n"
                "不要写成「用户倾向于详细规划和结构化思考。」这种带主语的整句，也不要\n"
                "把原话或事实抄一遍。\n"
                f"{_label_rule()}\n"
                '只输出 JSON：{"items": [{"slot": "喜好与厌恶", "label": "讨厌被打断"}, ...]}'
                '（items 可以是空列表 []）'
            )
        else:
            prompt = (
                f"The user said this (current emotion: {emotion or 'unknown'}):\n"
                f"\"{text[:300]}\"\n\n"
                "Does it reveal any of these subjective things about the speaker? "
                "At most ONE short label per category (3-8 words):\n"
                "- 喜好与厌恶: gut likes / dislikes / preferences\n"
                "- 表达风格: habits of speaking and communicating\n"
                "- 思维模式: how they think, weigh things, decide\n"
                "- 应对方式: what they do to cope with stress or bad feelings\n"
                "- 情绪: WHEN they feel WHAT. **A pattern, never a bare feeling "
                "word**: \"tense before design reviews\", \"annoyed when "
                "interrupted\", \"calm when alone\" — NOT \"anxious\" / \"happy\". "
                "It becomes the title of a node on a graph; a bare word says nothing.\n\n"
                "Skip any category the utterance does not clearly show.\n"
                "**How to write a label**: a short pattern, no subject, no full stop:\n"
                "  good: hates being interrupted / wants comfort under stress / "
                "conclusion first\n"
                "  bad: The user tends to plan in detail. (a full sentence with a subject)\n"
                "  bad: I major in computer science (copying the utterance / a plain fact)\n"
                "The slot names above are internal keys — keep them exactly as written, "
                "in Chinese. Only the label follows the language rule below.\n"
                f"{_label_rule()}\n"
                'Output JSON only: {"items": [{"slot": "喜好与厌恶", '
                '"label": "hates being interrupted"}, ...]} (items may be [])'
            )
        raw = self._llm_json(prompt)
        if not raw:
            return []
        try:
            items = _json.loads(raw).get("items", [])
        except Exception:
            return []
        valid_slots = {"喜好与厌恶", "表达风格", "思维模式", "应对方式", "情绪"}
        result = []
        for it in items:
            slot = str(it.get("slot", "")).strip()
            label = str(it.get("label", "")).strip()
            if slot in valid_slots and label:
                result.append((slot, label))
        return _keep(result)


    def _embed_text(self, text: str) -> list[float]:
        """Reuse or cache embeddings for graph entities, slot anchors and affective traits."""
        if self._embedder is not None:
            return self._embedder.embed_query_text(text) if hasattr(
                self._embedder, "embed_query_text") else self._embedder.embed_texts([text])[0]
        # Reuse the factual embedder's cached vector when it already encoded this exact text.
        from voicemem.utils.common import embed_cache
        model = resolve_model(role="embedding")
        return embed_cache.resolve(model, [text], self._embed_uncached)[0]


    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI
        client = OpenAI(
            api_key=resolve_api_key(),
            base_url=self._base_url,
            timeout=15.0,
        )
        _kw = {
            "model": resolve_model(role="embedding"),
            "input": texts,
            "encoding_format": "float",
        }
        if "openrouter" in str(resolve_base_url(self._base_url) or "").lower():
            _kw["extra_body"] = {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
        resp = client.embeddings.create(**_kw)
        _exp = int(os.environ.get("VOICEMEM_EMBED_DIM", "1536"))
        if len(resp.data[0].embedding) != _exp:
            raise RuntimeError(f"embedding 维度 {len(resp.data[0].embedding)} != {_exp}，供应商被换掉了")
        data = resp.data
        if all(d.index is not None for d in data):
            data = sorted(data, key=lambda d: d.index)
        return [list(map(float, row.embedding)) for row in data]


    def _llm_json(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=512,
            )
            from voicemem.utils.common.cost_log import log_usage
            log_usage("llm_json", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[SplitMgr] LLM 失败: {e}")
            return ""


    def _llm_text(self, prompt: str, max_tokens: int = 300) -> str:
        """Request plain-text output for summaries; return an empty string on provider failure."""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=15.0,
            )
            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
            )
            from voicemem.utils.common.cost_log import log_usage
            log_usage("llm_text", resp.model, getattr(resp, "usage", None))
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Attribution] LLM 失败: {e}")
            return ""


    # Factual retrieval delegates.

    def SearchCogGraph(self, *a, **k) -> tuple[set[str], QueryClassification]:
        """Delegate slot-based candidate selection to LeftBrain."""
        return self._left.SearchCogGraph(*a, **k)

    def SearchData(self, *a, **k) -> set[str]:
        """Delegate entity-based candidate refinement to LeftBrain."""
        return self._left.SearchData(*a, **k)

    def _search_data_impl(self, *a, **k) -> tuple[set[str], list[str]]:
        """Return candidate IDs and activated entity names from LeftBrain."""
        return self._left._search_data_impl(*a, **k)

    def _widen_for_time_question(self, *a, **k) -> set[str]:
        """Delegate temporal candidate expansion to LeftBrain."""
        return self._left._widen_for_time_question(*a, **k)

    def Rank(self, *a, **k) -> list[MemorySearchHit]:
        """Delegate factual vector ranking to LeftBrain."""
        return self._left.Rank(*a, **k)

    # Slot metadata delegates.

    def _get_slot_base_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_base_embeddings(*a, **k)

    def _get_slot_dyn_embeddings(self, *a, **k) -> dict[str, list[float]]:
        return self._left._get_slot_dyn_embeddings(*a, **k)

    def _normalize_slot_name(self, *a, **k) -> str:
        return self._left._normalize_slot_name(*a, **k)

    def _llm_tag_memories(self, *a, **k) -> list[str]:
        return self._left._llm_tag_memories(*a, **k)

    # Query classification and subgraph bookkeeping.

    @scoped_operation
    def Classify(self, *a, **k) -> QueryClassification:
        """Classify query slots and entities with the configured LeftBrain classifier."""
        return self._left.Classify(*a, **k)

    def PrimeSubgraphFromQuery(self, query: str, top_k: int = 10) -> dict:
        """Run classification and retrieval, returning the existing activation count."""
        classification = self.Classify(query)
        result = self.Search(
            query=query, slots=classification.slots, entities=classification.entities,
            top_k=top_k,
        )
        return {"status": "recorded", "count": len({h.memory_id for h in result.hits})}

    def _record_subgraph_activation(self, *a, **k) -> None:
        """Record retrieved candidates for session-level subgraph processing."""
        return self._left._record_subgraph_activation(*a, **k)

    def RunSubgraphCheckpoint(self, *a, **k) -> dict:
        """Delegate the current session's subgraph checkpoint to LeftBrain."""
        return self._left.RunSubgraphCheckpoint(*a, **k)

    def ArchiveColdMemories(self, *a, **k) -> dict:
        """Delegate cold-memory archiving to LeftBrain."""
        return self._left.ArchiveColdMemories(*a, **k)

    # Combined factual and affective retrieval.

    @scoped_operation
    def Search(
        self,
        query: str,
        slots: list[str] | None = None,
        entities: list[str] | None = None,
        emotion: str | None = None,
        top_k: int = 5,
        scene_filter: str | None = None,
        speaker_filter: str | None = None,
    ) -> SearchResult:
        """Retrieve combined factual and affective context.

        Apply the existing temporal and scene handling, construct factual candidates,
        then run vector ranking and affective retrieval concurrently. Supplied slots,
        entities, filters and reply context retain their current precedence and scope."""
        import time
        import concurrent.futures

        # Expand relative dates before classification and retrieval. Persistence normalizes observed dates to absolute values.
        from voicemem.leftbrain.time_expand import expand_relative_dates
        query = expand_relative_dates(query)

        # An explicit scene filter takes precedence over a scene inferred from the query.
        if scene_filter is None:
            from voicemem.utils.audio.environment.scene_classifier import infer_scene_from_text
            inferred_scene = infer_scene_from_text(query)
            if inferred_scene is not None:
                scene_filter = inferred_scene.value

        # Recent scene context is a soft preference; empty scene matches retain the wider candidate pool.
        if scene_filter is None:
            try:
                current_scene = self._get_trigger_store().get_last_scene(self._user_id)
                # An unknown scene must not narrow the candidate pool.
                if current_scene and current_scene != "unknown":
                    scene_filter = current_scene
            except Exception:
                pass

        # Build factual candidates and related summaries before either ranking branch starts.
        left = self._left.search(
            query, slots, entities, scene_filter, speaker_filter,
        )
        slot_mem_ids      = left["slot_mem_ids"]
        final_ids         = left["final_ids"]
        activated_names   = left["activated_names"]
        classification    = left["classification"]
        related_summaries = left["related_summaries"]
        t0, t1, t2 = left["t0"], left["t1"], left["t2"]

        # Vector ranking and affective retrieval can run concurrently once candidates and entity names are available.
        rb_hits: list[RightBrainHit] = []
        rb_directive = ""
        rb_duration  = 0.0

        # Use the preceding assistant reply for reaction context, not the reply to the current query.
        agent_reply = self.last_agent_reply()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            rb_future = pool.submit(
                contextualize(self._right.search), query, activated_names, emotion, top_k, agent_reply,
            )  # Affective branch starts concurrently.

            hits = self._left.rank(query, final_ids, top_k, speaker_filter=speaker_filter)
            t3 = time.time()

            rb_hits, rb_directive = rb_future.result()  # Wait after factual ranking completes.
            t4 = time.time()
            rb_duration = t4 - t2  # Wall time since affective candidates became available.

        # Emit abstention guidance only when neither branch supplies specific evidence for this query.
        _specific_rb = {"response_experience", "situation_pattern", "relation"}
        rb_specific = any(h.source in _specific_rb for h in rb_hits)
        left_weak = (not hits) or (not activated_names)
        if left_weak and not rb_specific:
            hint = (
                "Note: the memory system found no specific evidence for this query "
                "(only generic profile context). If the retrieved content does not "
                "actually answer the question, say you don't know instead of guessing."
                if _is_en_text(query) else
                "注意：记忆系统没有为该问题找到具体证据（只有泛化画像信息）。"
                "若检索内容不能真正回答问题，请直接说不知道，不要猜测。"
            )
            rb_directive = f"{rb_directive}\n{hint}".strip()

        # Render any scene-specific response directive separately from the retrieved memories.
        scene_directive = ""
        current_scene = ""
        try:
            from voicemem.utils.audio.environment.scene_classifier import SceneTag, scene_to_response_directive
            last_scene = self._get_trigger_store().get_last_scene(self._user_id)
            if last_scene:
                current_scene = last_scene
                try:
                    scene_directive = scene_to_response_directive(SceneTag(last_scene))
                except ValueError:
                    pass
        except Exception:
            pass

        # Record completed retrieval for session-level subgraph checks.
        self._record_subgraph_activation(hits)

        return SearchResult(
            hits=hits,
            classification=classification,
            related_summaries=related_summaries,
            slot_mem_ids=slot_mem_ids,
            final_candidate_ids=final_ids,
            search_mode=_search_mode(slot_mem_ids, final_ids),
            rb_directive=rb_directive,
            rb_hits=rb_hits,
            scene_directive=scene_directive,
            current_scene=current_scene,
            timing={
                "slot_filter_ms":    round((t1 - t0) * 1000, 1),
                "entity_narrow_ms":  round((t2 - t1) * 1000, 1),
                "rank_ms":           round((t3 - t2) * 1000, 1),
                "rb_ms":             round(rb_duration * 1000, 1),
                "total_ms":          round((t4 - t0) * 1000, 1),
            },
        )


    def _get_user_name(self) -> str | None:
        """Read the cached owner name through LeftBrain."""
        return self._left._get_user_name()

    @staticmethod
    def _is_english(text: str) -> bool:
        """Return whether CJK characters make up less than 30% of alphabetic characters."""
        cjk = sum(1 for c in text if "一" <= c <= "鿿" or "぀" <= c <= "ヿ")
        alpha = sum(1 for c in text if c.isalpha())
        return alpha > 0 and cjk / max(alpha, 1) < 0.3


    def _generate_inner_os(self, text: str, emotion: str, entities: list[str],
                           agent_reply: str = "") -> str:
        """Generate the existing third-person affective observation.

        Use the preceding assistant reply for reaction context and the captured
        operation language for the prompt. Return an empty string on failure."""
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=resolve_api_key(),
                base_url=self._base_url,
                timeout=10.0,
            )
            user_name = self._get_user_name()
            entity_hint = f", involving: {', '.join(entities)}" if entities else ""
            # Use the captured operation language, with the Space language as its fallback.
            from voicemem.lang import is_zh as _os_is_zh
            is_chinese = _os_is_zh()
            pronoun = user_name if user_name else ("用户" if is_chinese else "they")
            if is_chinese:
                system_prompt = (
                    f"你是一个有共情能力的AI助手，用第三人称记录你对用户情绪状态的内心感受。"
                    f"根据用户说的话，写出你（AI）的内心反应——就像你悄悄感受到了TA的情绪并被打动。"
                    f"要求：第三人称（称呼用户为『{pronoun}』），口语化，温暖，15-25字，"
                    f"开头用【情绪词】格式标注情绪。只输出一句话，不加任何解释。"
                    f"示例：\n"
                    f"输入：今天被老板当众批评了，好委屈\n"
                    f"输出：【心疼】{pronoun}强撑着没崩，但被这样当众说，心里一定很难受。\n"
                    f"输入：最好的朋友要搬走了\n"
                    f"输出：【担心】{pronoun}要失去身边最近的人了——以后难过的时候找谁说呢。"
                )
            else:
                system_prompt = (
                    "You are an empathetic AI assistant recording your inner observations about the user's emotional state. "
                    "Based on what the user said, write your (the AI's) internal reaction — "
                    "as if you quietly sensed their emotion and were moved by it. "
                    f"Requirements: third person (refer to the user as '{pronoun}'), "
                    "conversational, warm, 15-25 words, start with [emotion word] in brackets. "
                    f"Examples:\n"
                    f"Input: Got yelled at by my boss today, emotion: sad\n"
                    f"Output: [heartache] {pronoun} is holding it together on the outside, but being called out like that must really sting.\n"
                    f"Input: My best friend is moving away, emotion: longing\n"
                    f"Output: [worried] {pronoun} is losing someone close — once they're gone, who do they call on a hard day?\n"
                    "Output only that one sentence, nothing else."
                )
            reply_line = (agent_reply or "").strip()
            if reply_line:
                prior = ("你（AI）上一句说的是" if is_chinese else "What you (the AI) just said")
                user_content = (f"{prior}: {reply_line[:200]}\n"
                                f"What the user said: {text}\nEmotion: {emotion}{entity_hint}")
            else:
                user_content = f"What the user said: {text}\nEmotion: {emotion}{entity_hint}"

            resp = client.chat.completions.create(
                model=resolve_model(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=80,
                temperature=0.7,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""


    def _detect_scene(self, *a, **k):
        return self._audio._detect_scene(*a, **k)

    def _detect_speaker(self, *a, **k):
        return self._audio._detect_speaker(*a, **k)

    def _bind_self_identity(self, *a, **k):
        return self._audio._bind_self_identity(*a, **k)

    @scoped_operation
    def preprocess(self, *a, **k) -> "AudioPerception":
        """Delegate audio perception within the operation's captured language scope."""
        return self._audio.preprocess(*a, **k)


    def remember_reply(self, user_text: str, reply: str) -> None:
        """Register a completed assistant exchange, skipping an identical latest pair."""
        reply = (reply or "").strip()
        if not reply:
            return
        pair = ((user_text or "").strip(), reply)
        if self._exchanges and self._exchanges[-1] == pair:
            return  # Avoid registering the same completed exchange twice.
        self._exchanges.append(pair)

    def last_agent_reply(self, before_text: str | None = None) -> str:
        """Return the latest reply, optionally skipping the reply to before_text."""
        want = (before_text or "").strip()
        for user_text, reply in reversed(list(self._exchanges)):
            if want and user_text == want:
                continue  # Skip the reply to the current input when finding preceding context.
            return reply
        return ""

    def _reply_to(self, text: str) -> str:
        """Return the registered reply to this exact user text, or an empty string."""
        want = (text or "").strip()
        for user_text, reply in reversed(list(self._exchanges)):
            if user_text == want:
                return reply
        return ""

    @scoped_operation
    def Ingest(
        self,
        text: str,
        speaker: str = "Speaker 0",
        emotion: str = "",
        entities: list[str] | None = None,
        session_id: int | str | None = None,
        audio_path: str | None = None,
        observed_at: str | None = None,
        async_facts: bool = False,
        agent_reply: str | None = None,
        on_complete=None,
        *,
        language: str | None = None,
    ) -> dict:
        """Ingest an utterance using the existing perception and persistence pipeline.

        observed_at supplies the observation date; historical backfills should provide
        it explicitly. agent_reply disambiguates current facts, while the preceding
        reply supplies reaction context. async_facts=True schedules persistent writes
        in a captured worker context and returns acoustic metadata immediately.
        on_complete receives the durable result or error result on the worker thread;
        synchronous ingestion invokes it before returning. Callback errors are logged."""
        import time

        ts = observed_at or time.strftime("%H:%M:%S")

        if agent_reply is None:
            agent_reply = self._reply_to(text)  # Current reply disambiguates factual extraction.
        else:
            self.remember_reply(text, agent_reply)  # Register replies supplied by external generation.
        prior_reply = self.last_agent_reply(before_text=text)  # Preceding reply supplies reaction context.

        # Capture acoustic metadata before scheduling persistent writes.
        p = self.preprocess(text, speaker, emotion, session_id, audio_path)
        speaker          = p.speaker
        emotion          = p.emotion
        environment      = p.environment
        environment_hint = p.environment_hint
        scene_tag        = p.scene_tag
        scene_raw_labels = p.scene_raw_labels
        person_id        = p.person_id
        tune_result      = p.tune_result
        abnormal_hits    = p.abnormal_hits
        detection        = p.detection
        place_result     = None  # Filled during scene clustering.
        new_routine      = None  # Filled during routine detection.

        ctx = {
            "text": text, "speaker": speaker, "emotion": emotion, "entities": entities,
            "session_id": session_id, "audio_path": audio_path, "observed_at": observed_at,
            "ts": ts,
            # AST remains the immediate hint. When CLAP final-memory mode
            # is enabled, don't put that provisional text in the utterance memory.
            "environment": "" if self._clap_memory_enabled() else environment,
            "environment_hint": environment_hint,
            "scene_tag": scene_tag,
            "scene_raw_labels": scene_raw_labels, "person_id": person_id,
            "tune_result": tune_result, "abnormal_hits": abnormal_hits,
            "place_result": place_result, "new_routine": new_routine, "detection": detection,
            # Freeze current and preceding replies so later turns cannot change attribution context.
            "agent_reply": agent_reply or "", "prior_agent_reply": prior_reply or "",
        }

        if self._clap_memory_enabled() and audio_path is not None:
            threading.Thread(
                target=contextualize(self._finish_clap_environment),
                args=(audio_path, text, session_id, environment_hint),
                daemon=True,
            ).start()

        def _notify_complete(result: dict) -> None:
            if on_complete is None:
                return
            try:
                on_complete(result)
            except Exception as e:
                print(f"[ingest] on_complete 回调失败：{type(e).__name__}: {e}", flush=True)

        # Background fact writes return acoustic metadata immediately. Completion reports durable results separately.
        if async_facts:
            def _bg() -> None:
                """Persist the captured turn, logging failures and reporting an error through the completion callback."""
                try:
                    _notify_complete(self._finish_ingest(ctx))
                except Exception as e:
                    import traceback
                    print(f"[ingest] 这一轮没能入库（{type(e).__name__}: {e}）\n"
                          f"{traceback.format_exc()}", flush=True)
                    _notify_complete({"error": str(e), "persistent_memory_created": False})

            threading.Thread(target=contextualize(_bg), daemon=True).start()
            return {
                "facts_count":         None,
                "memory_ids":          [],
                "affect":              None,
                "triggered_reminders": [],
                "proactive_memories":  [],
                "current_scene":       scene_tag or "",
                "environment_hint":    environment_hint,
                "speaker_id":          person_id or "",
                "recognized_tune":     (
                    {"tune_id": tune_result.tune_id, "action": tune_result.action,
                     "heard_count": tune_result.heard_count}
                    if tune_result is not None else None
                ),
                "abnormal_sounds":     [l for l, _ in abnormal_hits],
                "recognized_place":    (
                    {"place_id": place_result.place_id, "action": place_result.action,
                     "visit_count": place_result.visit_count,
                     "previous_visit_at": place_result.previous_visit_at}
                    if place_result is not None else None
                ),
                "familiar_place_prompt": None,
                "new_routine":         None,
            }

        result = self._finish_ingest(ctx)
        _notify_complete(result)
        return result

    def _tag_memories(self, memory_ids, tags) -> None:
        """Write supported memory tags; skip stores without the tagging interface."""
        cog_store = self._get_repo()._cognitive_store
        if cog_store and hasattr(cog_store, "upsert_memory_tags"):
            for mid in memory_ids:
                cog_store.upsert_memory_tags(mid, self._user_id, tags)

    def _extract_and_append(self, messages, instructions, ts, extra_metadata):
        """Extract synthetic audio-event facts and append them, returning backend memory IDs."""
        extracted = self._get_extractor().extract(
            new_messages=messages, custom_instructions=instructions,
            observation_date=ts, current_date=ts,
        )
        if not extracted:
            return []
        return self._get_repo().append_extracted(
            extracted, user_id=self._user_id, extra_metadata=extra_metadata)

    def _finish_ingest(self, ctx: dict) -> dict:
        """Persist facts, affective context and audio metadata from the frozen ingestion context."""
        text = ctx["text"]; speaker = ctx["speaker"]; emotion = ctx["emotion"]
        entities = ctx["entities"]; session_id = ctx["session_id"]
        audio_path = ctx["audio_path"]; observed_at = ctx["observed_at"]
        ts = ctx["ts"]; environment = ctx["environment"]; scene_tag = ctx["scene_tag"]
        scene_raw_labels = ctx["scene_raw_labels"]; person_id = ctx["person_id"]
        tune_result = ctx["tune_result"]; abnormal_hits = ctx["abnormal_hits"]
        place_result = ctx["place_result"]; new_routine = ctx["new_routine"]
        detection = ctx["detection"]
        environment_hint = ctx.get("environment_hint", "")
        agent_reply = ctx.get("agent_reply", "")
        prior_reply = ctx.get("prior_agent_reply", "")

        import uuid
        from voicemem.utils.common.voice_input import VoiceInput, VoiceContent

        vi = VoiceInput(
            id=f"utt_{uuid.uuid4().hex[:8]}",
            time_stamp={"begin": ts, "end": ts},
            slots=[],
            contents=[VoiceContent(
                sub_id="0", time_start=ts, time_end=ts,
                sentence=text, voiceprint_id=speaker, emotion=emotion,
            )],
            environment=environment,
            agent_reply=agent_reply,  # Include assistant wording in extraction context.
        )

        # LeftBrain writes facts; the shared speaker registry supplies identity mappings.
        result = self._left.ingest_facts(
            vi,
            registry=self._get_registry(),
            session_id=session_id,
            extra_metadata={"created_at": observed_at} if observed_at else None,
        )

        # Store assistant wording verbatim with assistant attribution. Normal factual queries exclude it;
        # queries about earlier assistant replies can retrieve it explicitly.
        if agent_reply.strip():
            try:
                self._get_repo()._vector_store.add_text(
                    self._user_id, agent_reply.strip(), attributed_to="assistant",
                    metadata={"source": "agent_reply", "turn_id": vi.id,
                              **({"time_start": observed_at} if observed_at else {})},
                )
            except Exception as e:
                print(f"[ingest] 存助手原话失败：{e}", flush=True)

        # Sound-only recordings still need a memory ID so archived audio and replay tags have a persistent target.
        from voicemem.utils.audio.perceiver import said_music as _is_said_music
        _said_music = _is_said_music(text)
        _tune_id = getattr(tune_result, "tune_id", None) if tune_result else None

        # An announcement and its following audio can be separate turns.
        # A sound-only turn may inherit music intent; other sound-only turns retain a neutral sound label.
        from voicemem.stream import SOUND_ONLY_TEXT
        _sound_only = bool(audio_path) and (
            not (text or "").strip() or (text or "").strip() == SOUND_ONLY_TEXT)

        _expect = getattr(self, "_expect_music_until", 0.0)
        _inherited = _sound_only and time.monotonic() < _expect
        if _said_music:
            self._expect_music_until = time.monotonic() + EXPECT_MUSIC_S
        elif (text or "").strip():
            self._expect_music_until = 0.0  # A new utterance clears pending music intent.
        if _inherited:
            print("  [music] 上一轮说了要放音乐，这一轮只有声音 → 就是那首", flush=True)

        if not result.memory_ids and (_tune_id or _said_music or _inherited or _sound_only):
            try:
                heard = getattr(tune_result, "heard_count", 0) or 0
                again = "（之前也听过）" if heard > 1 else ""
                # Label audio as music only when recognition, explicit wording or inherited intent supports it.
                what = "音乐" if (_tune_id or _said_music or _inherited) else "声音"
                _content = f"用户放了一段{what}给我听{again}。"
                mid = self._get_repo()._vector_store.add_text(
                    self._user_id, _content,
                    metadata={"source": "sound_only", "turn_id": vi.id,
                              **({"tune_id": _tune_id} if _tune_id else {}),
                              **({"time_start": observed_at} if observed_at else {})},
                )
                if mid:
                    # Create cognitive metadata before adding tags: memory_tags references the memories table.
                    try:
                        from voicemem.leftbrain.cognitive_graph.slot_v2 import SlotV2
                        self._get_repo()._cognitive_store.upsert_memory_record(
                            self._user_id, mid, SlotV2.DAILY_LIFE, _content,
                        )
                        # sound_only distinguishes the recording from spoken announcements.
                        # tune:* makes music available to replay; unidentified preserves unknown track identity.
                        tags = [("sound_only", 0.95)]
                        if _tune_id:
                            tags.append((f"tune:{_tune_id}", 0.9))
                        elif _said_music or _inherited:
                            tags.append(("tune:unidentified", 0.9))
                        self._tag_memories([mid], tags)
                    except Exception as e:
                        print(f"[ingest] 音乐轮补写 memories 失败（标签会挂不上）：{e}",
                              flush=True)
                    result.memory_ids = list(result.memory_ids or []) + [mid]
            except Exception as e:
                print(f"[ingest] 存音乐轮失败：{e}", flush=True)

        # Persist audio tags, reminders and archive links.
        audiomem = self._write_audiomem_tags(
            result, scene_tag, scene_raw_labels, detection, audio_path,
            person_id, tune_result, abnormal_hits, ts, session_id, text,
        )
        triggered_reminders = audiomem["triggered_reminders"]
        proactive_memories = audiomem["proactive_memories"]
        familiar_place_prompt = audiomem["familiar_place_prompt"]
        place_result = audiomem["place_result"]
        new_routine = audiomem["new_routine"]

        self._write_left_brain(result, text)
        heartnote_id = self._write_right_brain(
            emotion, result, text, entities, observed_at, prior_reply)
        # Learn reaction traits independently of heartnote creation, including when emotion is empty.
        self._right.learn_from_reaction(
            text, emotion, entities, prior_reply,
            memory_id=(result.memory_ids[0] if result.memory_ids else None),
            observed_at=observed_at, heartnote_id=heartnote_id,
        )

        # Schedule affective cleanup through its existing threshold policy.
        threading.Thread(target=contextualize(self._check_and_cleanup), daemon=True).start()
        # Archive cleanup applies the configured retention policy outside the write result path.
        threading.Thread(target=contextualize(self._check_and_cleanup_audio), daemon=True).start()

        # Session bookkeeping and batched attribution.
        turn_info = self._get_session_tracker().record_turn(self._user_id, session_id)

        # Consolidation summarizes accumulated evidence. Count unique new memory IDs,
        # then flush pending entities at the threshold or a session boundary.
        try:
            tracker = self._get_session_tracker()
            for mid in list(getattr(result, "memory_ids", None) or []):
                tracker.touch(self._user_id, "rb_pending_memories", str(mid))
            if heartnote_id:
                tracker.touch(self._user_id, "rb_pending_memories", str(heartnote_id))
            n = tracker.count_touched(self._user_id, "rb_pending_memories")
            if n >= SHORT_TERM_MIN_MEMORIES or turn_info["session_changed"]:
                tracker.pop_touched(self._user_id, "rb_pending_memories")  # Clear the accumulated count.
                touched = tracker.pop_touched(self._user_id, "rb_entity_short")
                if touched:
                    self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] 短期归因失败: {e}")

        if turn_info["session_changed"]:
            self._run_session_boundary_batch()

        return {
            "facts_count":         result.facts_count,
            "memory_ids":          result.memory_ids,
            "persistent_memory_created": bool(result.memory_ids or heartnote_id),
            "affect":              result.affect,
            "triggered_reminders": triggered_reminders,
            "proactive_memories":  proactive_memories,
            "current_scene":       scene_tag or "",
            "environment_hint":    environment_hint,
            "speaker_id":          person_id or "",
            "speaker_name":        (
                self._get_registry().display_name(person_id) if person_id else speaker
            ),
            "recognized_tune":     (
                {"tune_id": tune_result.tune_id, "action": tune_result.action,
                 "heard_count": tune_result.heard_count}
                if tune_result is not None else None
            ),
            "abnormal_sounds":     [l for l, _ in abnormal_hits],
            "recognized_place":    (
                {"place_id": place_result.place_id, "action": place_result.action,
                 "visit_count": place_result.visit_count,
                 "previous_visit_at": place_result.previous_visit_at}
                if place_result is not None else None
            ),
            "familiar_place_prompt": familiar_place_prompt,
            "new_routine":         new_routine,
        }

    def _write_audiomem_tags(self, *a, **k) -> dict:
        """Delegate audio tags, reminders and archive links to AudioPerceiver."""
        return self._audio._write_audiomem_tags(*a, **k)

    def _write_left_brain(self, result, text) -> None:
        """Delegate factual slot and entity metadata writes to LeftBrain."""
        return self._left.write(result, text)

    def _write_right_brain(self, emotion, result, text, entities, observed_at,
                           agent_reply: str = "") -> str | None:
        """Write affective context using the preceding reply; return the heartnote ID if created."""
        return self._right.write(emotion, result, text, entities, observed_at, agent_reply)

    def _run_session_boundary_batch(self) -> None:
        """Process session subgraphs, changed-slot summaries and long-term attribution.

        A changed session ID triggers this during ingestion. Flush() explicitly handles
        the final session, which has no following ingestion to trigger the boundary."""
        try:
            self.RunSubgraphCheckpoint()
        except Exception as e:
            print(f"[Subgraph] session边界判定失败: {e}")

        # Refresh summaries for slots whose memory counts changed during this session.
        try:
            self._refresh_schema_descriptions()
        except Exception as e:
            print(f"[SchemaDesc] 刷新失败: {e}")

        try:
            touched_slots = self._get_session_tracker().pop_touched(self._user_id, "rb_slot_long")
            if touched_slots:
                self._get_attribution_manager().run_long_term(self._user_id, touched_slots)
        except Exception as e:
            print(f"[Attribution] 长期归因失败: {e}")

    def _refresh_schema_descriptions(self) -> None:
        """Refresh changed-slot summaries through LeftBrain."""
        return self._left._refresh_schema_descriptions()

    @scoped_operation
    def Flush(self) -> None:
        """Process pending final-session attribution; no pending references means no attribution work."""
        self._run_session_boundary_batch()
        try:
            touched = self._get_session_tracker().pop_touched(self._user_id, "rb_entity_short")
            if touched:
                self._get_attribution_manager().run_short_term(self._user_id, touched)
        except Exception as e:
            print(f"[Attribution] 短期归因失败: {e}")

    def IngestEnv(self, *a, **k) -> dict:
        """Delegate an environmental sound event to AudioPerceiver."""
        return self._audio.IngestEnv(*a, **k)

    def _check_and_cleanup(self) -> None:
        """Delegate threshold-based affective cleanup to RightBrain."""
        return self._right.check_and_cleanup()

    def _check_and_cleanup_audio(self, retention_days: int = 30) -> None:
        """Apply recording retention at most once per day using the existing archive policy."""
        try:
            import json as _json
            from datetime import datetime, timezone
            last_run = _space.kv_get(self._memory_root, "audio_cleanup_last_run", "")
            now = datetime.now(timezone.utc)
            if last_run:
                try:
                    elapsed_hours = (now - datetime.fromisoformat(last_run)).total_seconds() / 3600
                except ValueError:
                    elapsed_hours = 999
            else:
                elapsed_hours = 999

            if elapsed_hours < 24:
                return

            _space.kv_set(self._memory_root, "audio_cleanup_last_run", now.isoformat())
            self._get_audio_archive().cleanup_expired(retention_days=retention_days)
        except Exception as e:
            print(f"[Cleanup] audio check error: {e}")

    def _run_cleanup(self) -> None:
        """Delegate affective memory cleanup to RightBrain."""
        return self._right.run_cleanup()



__all__ = ["Orchestrator", "SearchResult", "Utils"]
