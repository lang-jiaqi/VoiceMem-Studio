"""Local three-way reply routing after final ASR."""
from __future__ import annotations

import asyncio
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

FAST = "fast"
MEDIUM = "medium"
SLOW = "slow"
THINKING_LEVELS = (FAST, MEDIUM, SLOW)

_ROUTE_LABEL = re.compile(
    r"(即时|记忆|深思|fast|medium|slow)", re.IGNORECASE)
_SLOW_FLOOR = re.compile(
    r"(?:求|计算|算|解).{0,18}(?:积分|导数|微分方程)"
    r"|(?:积分|导数|微分方程).{0,10}(?:怎么求|怎么算|推导|过程)"
    r"|(?:证明|逐步推导)"
    r"|\b(?:solve|calculate|derive|prove).{0,32}"
    r"(?:integral|derivative|differential equation|proof)\b",
    re.IGNORECASE,
)
_FAST_FLOOR = re.compile(
    r"^(?:你好|您好|哈[喽啰罗]|嗨|早上好|中午好|下午好|晚上好|晚安|谢谢|再见)"
    r"|(?:介绍一下你自己|你是谁)"
    r"|(?:讲|说|编|写).{0,24}(?:故事|笑话)"
    r"|^(?:hi|hello|hey|thanks|thank you|good morning|good evening|goodbye)\b"
    r"|(?:introduce yourself|who are you|tell me .{0,24}(?:story|joke))",
    re.IGNORECASE,
)

_ZH_SYSTEM = """你是中文语音助手的回复路由器。根据用户最后一句话，只输出下列一个标签，
不要解释：

即时：仅凭当前消息和当前对话即可回答。包括寒暄、让助手介绍自己、普通事实、创作故事、
一般知识解释、简单建议和没有要求求解的数学名词。
记忆：答案必须依赖这个用户在过去会话中的个人事实、偏好、经历或安排；检索后可以直接回答。
出现“我、自己、你”不代表需要记忆，让助手介绍自己也不需要用户记忆。
深思：用户明确要求非平凡计算、积分求解、证明或逐步推导，或者要求同时权衡多个约束、
诊断复杂代码、制定多阶段方案。只有话题听起来专业、出现“为什么/解释/故事”，不能选深思。

先问：是否明确需要多步推理？否则再问：没有用户的长期记忆能否正确回答？能就选即时，
不能才选记忆。拿不准时选择即时。"""

_ZH_EXAMPLES = [
    ("你好呀。", "即时"),
    ("请介绍一下你自己。", "即时"),
    ("讲一个关于流浪猫的故事。", "即时"),
    ("解释一下为什么天空是蓝色的。", "即时"),
    ("sin x", "即时"),
    ("什么是积分？", "即时"),
    ("为什么规律作息能改善精神状态？", "即时"),
    ("法国的首都是什么？", "即时"),
    ("我上次说最喜欢什么？", "记忆"),
    ("我明天有什么安排？", "记忆"),
    ("求这个函数的不定积分，并写出推导过程。", "深思"),
    ("证明这个结论并逐步检查推导。", "深思"),
    ("结合我以前说过的预算，比较三个方案的风险和收益。", "深思"),
]

_EN_SYSTEM = """Route the user's final utterance. Output exactly one label and
nothing else: fast, medium, or slow.

fast: answer from the current message and conversation. Greetings, assistant
self-introduction, stories, ordinary explanations, simple advice, and math terms
without a request to solve them are fast.
medium: the answer requires personal facts, preferences, events, or plans from
the user's past sessions. Pronouns alone do not make a request medium.
slow: the user explicitly requests a non-trivial calculation, integral solution,
proof, derivation, complex debugging, multi-constraint comparison, or multi-stage
plan. A professional topic, a why/explain question, or a story is not slow.

Use slow only for explicit multi-step work, then medium only when personal memory
is necessary. When uncertain, choose fast."""

_EN_EXAMPLES = [
    ("Hello.", "fast"),
    ("Introduce yourself.", "fast"),
    ("Tell me a story about a stray cat.", "fast"),
    ("Explain why the sky is blue.", "fast"),
    ("Why does regular sleep improve energy?", "fast"),
    ("What did I say my favorite food was?", "medium"),
    ("What is on my schedule tomorrow?", "medium"),
    ("Solve this integral and show the derivation.", "slow"),
    ("Compare these plans across budget, risk, and benefit.", "slow"),
]


@dataclass(frozen=True)
class ThinkingDecision:
    """Normalized three-way reply route selected for one confirmed user turn."""

    level: str
    raw: str = ""

    @property
    def reasoning_effort(self) -> str:
        # Memory retrieval is not chain-of-thought. Only the slow route enables it.
        return {FAST: "none", MEDIUM: "none", SLOW: "high"}[self.level]

    @property
    def reply_mode(self) -> str:
        return {FAST: "direct", MEDIUM: "memory", SLOW: "memory_cot"}[self.level]

    @property
    def display_name(self) -> str:
        return {FAST: "instant", MEDIUM: "mem", SLOW: "mem+cot"}[self.level]


def parse_level(output: str, fallback: str = FAST) -> ThinkingDecision:
    """Parse a model label, returning an explicit safe fallback on bad output."""
    match = _ROUTE_LABEL.search(output or "")
    labels = {
        "即时": FAST,
        "记忆": MEDIUM,
        "深思": SLOW,
        "fast": FAST,
        "medium": MEDIUM,
        "slow": SLOW,
    }
    level = labels.get(match.group(1).lower(), fallback) if match else fallback
    if level not in THINKING_LEVELS:
        level = FAST
    return ThinkingDecision(level=level, raw=(output or "").strip())


class QwenThinkingRouter:
    """Lazy Qwen3-0.6B three-way reply router with serialized Torch inference.

    Chinese input receives a fully Chinese policy and examples. The model itself
    runs in non-thinking mode and emits one short label. Calls are cached by final
    ASR text so speculative and confirmed paths can share a decision.
    """

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        local = root / "models" / "reply-router" / "Qwen3-0.6B"
        self.model_name = (
            model
            or os.environ.get("VOICEMEM_THINKING_ROUTER_MODEL")
            or (str(local) if (local / "config.json").is_file() else "Qwen/Qwen3-0.6B")
        )
        self.device_name = device or os.environ.get("VOICEMEM_THINKING_ROUTER_DEVICE", "")
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_lock = threading.Lock()
        self._cache: dict[tuple[str, bool], ThinkingDecision] = {}

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model, self._device
        with self._load_lock:
            if self._model is not None:
                return self._tokenizer, self._model, self._device
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device_name or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=dtype, low_cpu_mem_usage=True)
            model.to(device).eval()
            self._tokenizer, self._model, self._device = tokenizer, model, device
            print(f"[thinking] router ready: {self.model_name} on {device}", flush=True)
        return self._tokenizer, self._model, self._device

    def _predict(self, system: str, examples, prompt: str) -> str:
        tokenizer, model, device = self._load()
        messages = [{"role": "system", "content": system}]
        for example, label in examples:
            messages.extend((
                {"role": "user", "content": example},
                {"role": "assistant", "content": label},
            ))
        messages.append({"role": "user", "content": prompt})
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(rendered, return_tensors="pt").to(device)
        import torch

        from voicemem.utils.torch_lock import TORCH_LOCK

        with TORCH_LOCK, torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return output.strip()

    @staticmethod
    def _is_chinese(text: str) -> bool:
        return any("\u3400" <= char <= "\u9fff" for char in text)

    def classify(self, text: str, memory_relevant: bool = False) -> ThinkingDecision:
        """Classify confirmed ASR text synchronously; callers run this off-loop."""
        key = (text, memory_relevant)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        def remember(decision: ThinkingDecision) -> ThinkingDecision:
            if len(self._cache) >= 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = decision
            return decision

        # These explicit math forms are product policy, not a topic heuristic.
        # Keeping the floor deterministic prevents a small router from turning a
        # requested derivation into an instant answer.
        if _SLOW_FLOOR.search(text):
            return remember(ThinkingDecision(SLOW, "policy-floor"))
        if _FAST_FLOOR.search(text):
            return remember(ThinkingDecision(FAST, "policy-floor"))
        if memory_relevant:
            return remember(ThinkingDecision(MEDIUM, "memory-gate"))
        compact = "".join(char for char in text if char.isalnum())
        if len(compact) <= 4:
            return remember(ThinkingDecision(FAST, "short-fragment"))

        chinese = self._is_chinese(text)
        output = self._predict(
            _ZH_SYSTEM if chinese else _EN_SYSTEM,
            _ZH_EXAMPLES if chinese else _EN_EXAMPLES,
            text.strip(),
        )
        fallback = MEDIUM if memory_relevant else FAST
        decision = parse_level(output, fallback)
        if not _ROUTE_LABEL.search(output or ""):
            print(f"[thinking] invalid router output {output!r}; fallback={fallback}",
                  flush=True)
        return remember(decision)

    async def classify_async(self, text: str, memory_relevant: bool = False) -> ThinkingDecision:
        """Run model inference outside the asyncio/WebSocket thread."""
        return await asyncio.to_thread(self.classify, text, memory_relevant)

    def warmup(self) -> ThinkingDecision:
        """Load weights and compile the short non-thinking generation path."""
        return self.classify("你好", False)


_ROUTER: QwenThinkingRouter | None = None


def thinking_router() -> QwenThinkingRouter:
    """Return the process-level lazy router shared by all Studio sessions."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = QwenThinkingRouter()
    return _ROUTER
