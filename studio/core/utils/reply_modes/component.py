"""Local three-way reply routing after final ASR."""
from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from studio.harness.reply_modes.policy import SYSTEM, EXAMPLES

FAST = "fast"
MEDIUM = "medium"
SLOW = "slow"
THINKING_LEVELS = (FAST, MEDIUM, SLOW)
_DEFAULT_ROUTER_REPO = "Qwen/Qwen3-0.6B"

def _local_router_ready(path: Path) -> bool:
    """Return whether a local snapshot has config, tokenizer, and weights."""
    has_tokenizer = any((path / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "vocab.json"))
    has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    return (path / "config.json").is_file() and has_tokenizer and has_weights

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
_MEMORY_FLOOR = re.compile(
    r"(?:还记得|你记得|我(?:上次|上回|以前|之前)(?:说|提|聊|告诉|答应)过)"
    r"|(?:上次|上回|以前|之前).{0,20}(?:说过|提过|聊过|喜欢|偏好|计划|决定)"
    r"|(?:今天|明天|后天|周末|下周|下个月).{0,8}(?:安排|日程|计划|有约)"
    r"|\b(?:remember|last time|previously|my schedule|my plans)\b",
    re.IGNORECASE,
)
_FOLLOWUP = re.compile(
    r"(?:这个|那个|那这个|然后|接着|继续|为什么|刚才|前面|上一个|比较一下|相比)"
    r"|(?:那.{0,12}呢|还有呢|然后呢)"
    r"|\b(?:this one|that one|continue|go on|why|earlier|previous one|compare)\b",
    re.IGNORECASE,
)



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

def _router_download_progress_class():
    """Return a lazy tqdm class whose output survives concise Studio logging."""
    from tqdm.auto import tqdm

    class RouterDownloadProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.update(
                desc="[status] Router 下载",
                disable=False,
                dynamic_ncols=True,
                file=sys.stdout,
                unit="文件",
            )
            super().__init__(*args, **kwargs)

    return RouterDownloadProgress

class QwenThinkingRouter:
    """Lazy Qwen3-0.6B three-way reply router with serialized Torch inference.

    All input uses one policy and example bank. The model itself
    runs in non-thinking mode and emits one short label. Calls are cached by final
    ASR text, bounded recent history, and the prefetch hint so speculative and
    confirmed paths share only context-compatible decisions.
    """

    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        from studio.paths import MODELS
        local = MODELS / "reply-router/Qwen3-0.6B"
        configured = model
        self._local_model_dir = local
        self._download_default = not configured and not _local_router_ready(local)
        self.model_name = configured or str(local)
        self.device_name = device or ''
        self._tokenizer = None
        self._model = None
        self._device = None
        self._load_lock = threading.Lock()
        self.history_messages = 4
        self.history_chars = 320
        self._cache: dict[tuple[str, bool, str], ThinkingDecision] = {}

    def _ensure_model_source(self) -> str:
        """Download the default router with visible progress when it is absent."""
        if not self._download_default:
            return self.model_name

        from huggingface_hub import snapshot_download

        destination = self._local_model_dir
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"[status] 三级回复路由缺失，开始下载 {_DEFAULT_ROUTER_REPO} "
              f"→ {destination}", flush=True)
        snapshot_download(
            repo_id=_DEFAULT_ROUTER_REPO,
            local_dir=str(destination),
            tqdm_class=_router_download_progress_class(),
        )
        if not _local_router_ready(destination):
            raise FileNotFoundError(f"router download incomplete: {destination}")
        self.model_name = str(destination)
        self._download_default = False
        print(f"[status] 三级回复路由下载完成：{destination}", flush=True)
        return self.model_name

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model, self._device
        with self._load_lock:
            if self._model is not None:
                return self._tokenizer, self._model, self._device
            self._ensure_model_source()
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

    def _context_prompt(self, text: str, history, prefetch_hint: bool) -> str:
        role_names = {"user": "用户", "assistant": "助手"}
        remaining = max(0, getattr(self, "history_chars", 320))
        count = max(0, getattr(self, "history_messages", 4))
        selected = list(history or [])[-count:] if count else []
        lines = []
        for message in reversed(selected):
            if remaining <= 0:
                break
            content = " ".join(str(message.get("content") or "").split())
            if not content:
                continue
            content = content[:remaining]
            remaining -= len(content)
            role = role_names.get(str(message.get("role")), "上下文")
            lines.append(f"{role}: {content}")
        lines.reverse()
        history_text = "\n".join(lines) if lines else "无"
        return (f"最近对话：\n{history_text}\n"
                f"当前用户：{text.strip()}\n"
                f"记忆预取提示：{'是' if prefetch_hint else '否'}")

    def classify(self, text: str, memory_prefetch_hint: bool = False,
                 history=None) -> ThinkingDecision:
        """Classify confirmed ASR text synchronously; callers run this off-loop."""
        prompt = self._context_prompt(text, history, memory_prefetch_hint)
        key = (text, memory_prefetch_hint, prompt)
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
        if _MEMORY_FLOOR.search(text):
            return remember(ThinkingDecision(MEDIUM, "policy-floor"))
        compact = "".join(char for char in text if char.isalnum())
        if len(compact) <= 4 and not (_FOLLOWUP.search(text) and history):
            return remember(ThinkingDecision(FAST, "short-fragment"))

        output = self._predict(
            SYSTEM,
            EXAMPLES,
            prompt,
        )
        fallback = MEDIUM if memory_prefetch_hint else FAST
        decision = parse_level(output, fallback)
        if not _ROUTE_LABEL.search(output or ""):
            print(f"[thinking] invalid router output {output!r}; fallback={fallback}",
                  flush=True)
        return remember(decision)

    async def classify_async(self, text: str, memory_prefetch_hint: bool = False,
                             history=None) -> ThinkingDecision:
        """Run model inference outside the asyncio/WebSocket thread."""
        return await asyncio.to_thread(
            self.classify, text, memory_prefetch_hint, history)

    def warmup(self) -> ThinkingDecision:
        """Load weights and compile the short non-thinking generation path."""
        # Use an intentionally ambiguous utterance so policy floors cannot skip
        # the model load and push a multi-second cold start onto the first user.
        return self.classify("介绍一下向量数据库", False)

