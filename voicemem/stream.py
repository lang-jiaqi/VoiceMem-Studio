"""voicemem 核心流式输入会话：边听边投机预取（EOU 0–300ms）。

和「文本」「wav」并列的第三种输入途径。两种喂法，每块都返回一个 ``StreamState``
（这块 ``<speak>``/``<silence>``、说完那块是 ``turn_over`` + 投机预取的记忆 + ``Turn``）：

    stream = vm.stream(on_partial=lambda t: print(t))

    # ① 喂音频块：voicemem 自带流式 ASR（FunASR paraformer）+ silero VAD
    st = await stream.feed(pcm_bytes)          # PCM16 @ src_rate（默认 24k）

    # ② 喂【外部 ASR】的 partial 文本（FunASR / Whisper / 任意）——换 ASR 只改喂进来的这行
    st = await stream.feed_partial(text, ended=is_final)

    st.state    # "<speak>" | "<silence>" | "turn_over"（这一轮说完了）
    st.memory   # 当前投机预取的记忆（SearchResult）；边说边有，没算好时 None
    st.turn     # 一轮说完才有 Turn（否则 None）

**只到记忆结果**——回复（tts/realtime）由调用方拿到 Turn/memory 后自理，核心不碰。
"""
from __future__ import annotations

import asyncio
import os
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voicemem import gate as _gate_mod
from voicemem.memory_api import build_memory_context
from voicemem.utils.audio.stream_io import resample


def empty_result():
    """"这一轮不需要记忆"时交出去的空结果。

    不用 ``None``：调用方（demo 的 hits_payload / note_hits / 回放挑选…）拿到的
    一直是 SearchResult，突然变成 None 就是一串 AttributeError。空结果让"没检索"
    和"检索了但一条都没有"走同一条代码路径。
    """
    from voicemem.leftbrain.cognitive_graph.query_slot_classifier import QueryClassification
    from voicemem.orchestrator import SearchResult
    return SearchResult(hits=[], classification=QueryClassification(slots=[], entities=[]),
                        related_summaries={}, slot_mem_ids=set(),
                        final_candidate_ids=set(), search_mode="gated")


@dataclass
class Turn:
    """一轮说完（或打字）时、投机预取早已算好的记忆结果——调用方拿来直接回复，不再搜。"""
    text: str
    result: object
    #: 复核之前的流式转写（见 StreamState.raw_text）。
    raw_text: str = ""
    #: 轮次闸门判的路（见 voicemem/gate.py）。``deep`` = 检索过了；其余两路没检索，
    #: ``result`` 是空的。调用方据此决定提不提示"我不知道"——浅轮上提那句，
    #: "讲个笑话"会被答成"我不知道"。
    route: str = _gate_mod.DEEP

    @property
    def memory_context(self) -> str:
        return build_memory_context(self.result)


@dataclass
class StreamState:
    """每喂一块（音频或外部 ASR 文本）返回：这块静音/说话 + 当前投机记忆 + 说完了没。

    下面那组感知字段（``emotion`` / ``speaker_id`` / ``speaker_voiceprint`` /
    ``entity`` / ``schema`` / ``text_embedding``）全是**取用时才算**的 property：
    不读就一分钱一毫秒都不花，投机预取那条 0–300ms 的路径完全不受影响。
    """
    state: str                 # "<speak>" | "<silence>" | "turn_over"（一轮说完）
    text: str                  # 到目前为止的累积转写
    memory: object | None      # 当前投机预取到的记忆（SearchResult）；没算好时 None
    turn: Turn | None          # 一轮说完才有，否则 None
    _vm: object = None         # 惰性感知要用到的能力（utils）；调用方不用管
    _pcm: object = None        # 本轮 16k 单声道音频，供声纹/情绪按需分析
    #: 这一刻已经静了多久（秒）。调用方拿它做「说到一半的停顿」判断——
    #: 附和（backchannel）就靠它：人停 100ms 时"嗯"一声，比等一轮说完再回应自然。
    #: 内部那两个门槛（gamble_s 200ms 赌说完 / confirm_s 300ms 确认）用的是同一个量。
    silence: float = 0.0
    #: 这一轮用户已经开过口了吗。没开过口时的静音是「还没开始」，不是「停顿」。
    spoke: bool = False
    #: 复核**之前**的那份流式转写。
    #:
    #: 调用方要拿"说话中途的文本"跟"说完时的文本"比（提前生成就靠它判赌没赌对），
    #: 而中途只有流式那份。拿它去比复核后的文本，等于拿两个模型的输出对齐——用词
    #: 永远不一样，比出来的差异是模型分歧，不是"他又说了话"。
    raw_text: str = ""
    #: 最近一次 EOT 打分（0~1，越大越像「这句话齐了」）。没启用 EOT 时是 0。
    #:
    #: **说话期间也在算**，不只静音时。用途是两个，阈值不一样：
    #:   判回合结束  要 silence + 高分（只看分会在句子中途的完整点切断）
    #:   提前起跑生成 只看分就够——赌错了取消重来，代价是钱不是体验
    eot_score: float = 0.0
    speech_end: float = 0.0  # Server monotonic time of the last voiced input frame.

    @property
    def memory_context(self) -> str:
        m = self.turn.result if self.turn else self.memory
        return build_memory_context(m) if m is not None else ""

    @property
    def route(self) -> str:
        """这一轮闸门判的路。没说完那一轮还没最终判定，按 deep 报。"""
        return self.turn.route if self.turn else _gate_mod.DEEP

    # ── 记忆结果 ───────────────────────────────────────────────────────────────

    @property
    def _result(self):
        return self.turn.result if self.turn else self.memory

    @property
    def transcript(self) -> str:
        return self.text

    @property
    def result_leftbrain(self) -> list[str]:
        r = self._result
        return list(r.result_leftbrain) if r is not None else []

    @property
    def result_rightbrain(self) -> list[str]:
        r = self._result
        return list(r.result_rightbrain) if r is not None else []

    @property
    def entity(self) -> list[str]:
        """这句话里的命名实体（投机预取时已经分类过，直接取，不重算）。"""
        r = self._result
        return list(getattr(r.classification, "entities", []) or []) if r is not None else []

    @property
    def schema(self) -> list[str]:
        """这句话路由到的记忆槽位。"""
        r = self._result
        return list(getattr(r.classification, "slots", []) or []) if r is not None else []

    # ── 声学感知（取用时才算）───────────────────────────────────────────────────

    @property
    def _perception(self):
        if getattr(self, "_p_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._p_cache = _perceive(self._vm, self._pcm, self.text)
        return self._p_cache

    @property
    def emotion(self) -> str:
        p = self._perception
        return getattr(p, "emotion", "") if p else ""

    @property
    def speaker_id(self) -> str:
        p = self._perception
        return (getattr(p, "person_id", None) or "") if p else ""

    @property
    def speaker_voiceprint(self):
        """本轮说话人的声纹向量（numpy 数组）；没开声纹或算不出时 None。"""
        if getattr(self, "_vp_cache", None) is None:
            if self._vm is None or self._pcm is None or not len(self._pcm):
                return None
            self._vp_cache = _voiceprint(self._vm, self._pcm)
        return self._vp_cache

    @property
    def text_embedding(self):
        """这句转写的文本向量；没有文本时 None。"""
        if getattr(self, "_emb_cache", None) is None:
            if self._vm is None or not self.text.strip():
                return None
            self._emb_cache = _embed(self._vm, self.text)
        return self._emb_cache


# ── StreamState 那几个感知字段的实现（都只在被读到时才跑）──────────────────────

def _tmp_wav(pcm) -> str:
    """把本轮音频落成临时 wav——声纹/情绪模块都按文件路径取音频。"""
    import tempfile, wave
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _perceive(vm, pcm, text):
    """跑一次音频感知（场景/声纹/情绪），复用编排层现成的 preprocess，不另起一套。"""
    path = _tmp_wav(pcm)
    try:
        return vm.preprocess(text or "", audio=path)
    except Exception as e:
        print(f"[stream] 感知失败: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass


def _voiceprint(vm, pcm):
    path = _tmp_wav(pcm)
    try:
        return vm.utils.get("voiceprint").embed(Path(path))
    except Exception as e:
        print(f"[stream] 声纹提取失败: {e}", flush=True)
        return None
    finally:
        import os
        try: os.unlink(path)
        except OSError: pass


def _embed(vm, text):
    try:
        emb = vm.utils.get("embedding")
        fn = getattr(emb, "embed_texts", None)
        return fn([text])[0] if fn else emb.embed(text)
    except Exception as e:
        print(f"[stream] 文本向量失败: {e}", flush=True)
        return None


#: 每轮最多留这么长的音频给按需感知用（16k 单声道，30s ≈ 1.9MB）
_MAX_TURN_SAMPLES = 30 * 16000

#: 开口前保留这么长的音频，免得切掉第一个字（16k 单声道）
_PREROLL_SAMPLES = int(0.3 * 16000)

#: 没有转写文本、但持续这么久的声音也算一轮（对着麦克风放音乐的场景）。
#: 设长一点：短促的环境噪声（关门、咳嗽）不该变成一轮。
MIN_SOUND_ONLY_S = float(os.environ.get("VOICEMEM_SOUND_ONLY_S", "5"))

#: 纯声音的一轮，要静多久才算结束。
#:
#: 说话那一轮用 confirm_s（300ms）——对话就该这么快。但音乐不是对话：乐句之间的
#: 停顿、弱拍、前奏后的留白，随便就超过 300ms，于是一首歌被切成一地碎片（实测
#: 归档的录音全是 1.5~7.8 秒），回放时放出来只有开头几秒。
#: 放音乐的场景本来也不需要秒回，等久一点换一段完整的录音，划算。
SOUND_ONLY_SILENCE_S = float(os.environ.get("VOICEMEM_SOUND_ONLY_SILENCE_S", "3.0"))

#: 这一块音频有没有声音（RMS 阈值）。
#:
#: 判"一轮结束"平时看 VAD，而 silero VAD 判的是**有没有人声**——音乐不是人声，
#: 所以整段音乐在它眼里都是静音，静音计数一路涨，到 SOUND_ONLY_SILENCE_S 就把
#: 这轮切断了。实测放一首歌，归档的录音只有 3.0 秒，正好等于那个阈值。
#: 所以一个字都没转出来的时候改看能量：还有声音就不算静音，音乐放多久录多久。
SOUND_LEVEL = float(os.environ.get("VOICEMEM_SOUND_LEVEL", "0.01"))
#: 流式 ASR 的音频积压超过这么多秒就报一行（限频）。
ASR_BACKLOG_WARN_S = float(os.environ.get("VOICEMEM_ASR_BACKLOG_WARN", "0.4"))
ASR_DEBUG = os.environ.get("VOICEMEM_ASR_DEBUG", "0") == "1"
#: 投机检索节流：距上次起跑不到这么久、且文本没多出 SPEC_MIN_GROWTH 个字，就不重跑。
#: 实测一轮里跑 3 次、每次 300~600ms，全在 CPU 上跟流式 ASR 抢——ASR 每块从 55ms
#: 涨到 167ms，字就一顿一顿地出。检索只对"说完那一刻"有用，中途那几次多半白算。
SPEC_MIN_INTERVAL_S = float(os.environ.get("VOICEMEM_SPEC_MIN_INTERVAL", "0.8"))
SPEC_MIN_GROWTH = int(os.environ.get("VOICEMEM_SPEC_MIN_GROWTH", "6"))
#: 闸门判定和投机检索都要过 embedding（torch MPS），两条线程池线程同时提交 MPS
#: 会撞 Metal 断言直接 abort。串起来：同一时刻只有一个在算。
from voicemem.utils.torch_lock import TORCH_LOCK as _EMBED_LOCK   # 进程级 torch 锁
# One shared recognizer may serve multiple sessions; never decode it concurrently.
# Separate from the default executor so ingestion cannot queue ahead of final ASR.
_FINAL_ASR_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr-final")


class _AsrWorker:
    """把流式 ASR 挪到自己的串行线程上。

    原来 ``feed()`` 里是 ``self._text = asr.feed(frame)`` **同步**调：paraformer 攒够
    600ms 就在事件循环里推一块，一块几十到几百毫秒——这期间 socket 不读、音频不发、
    VAD/EOT 全停。机器一紧张（换页）就是"字一顿一顿地出、回复也跟着卡"。

    现在 ``push()`` 只入队、立刻返回最近一次识别出的累积文本；识别在这条线程上串行
    跑（模型本身不能并发）。``flush()`` 返回一个 Future：把队里剩的都推完、再跑
    ``is_final``，调用方 ``await`` 它拿最终文本。

    顺带量两件事，都是原来看不见的：每块推理用了多久、队里积了多少秒音频还没识别。
    积压说明 ASR 跟不上实时——那是机器的问题不是代码的问题，但得先量出来。
    """

    def __init__(self, asr):
        self.asr = asr
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.text = ""
        self.epoch = 0           # reset 后的旧推理结果不可污染下一轮
        self.backlog = 0          # 队里还没识别的样本数（16k）
        self.stats = {"chunks": 0, "busy_s": 0.0, "max_backlog": 0, "last_ms": 0.0}
        self._last_warn = 0.0
        self.t = threading.Thread(target=self._run, name="asr-worker", daemon=True)
        self.t.start()

    def push(self, frame) -> str:
        with self.lock:
            self.backlog += len(frame)
            self.stats["max_backlog"] = max(self.stats["max_backlog"], self.backlog)
            backlog_s, text = self.backlog / 16000.0, self.text
            self.q.put(("feed", frame, None, self.epoch))
        now = time.monotonic()
        if backlog_s >= ASR_BACKLOG_WARN_S and now - self._last_warn > 2.0:
            self._last_warn = now
            print(f"[asr] 积压 {backlog_s*1000:.0f}ms 音频没识别"
                  f"（上一块推理 {self.stats['last_ms']:.0f}ms）→ ASR 跟不上实时",
                  flush=True)
        return text

    def flush(self):
        fut = asyncio.get_running_loop().create_future()
        loop = asyncio.get_running_loop()
        with self.lock:
            self.q.put(("flush", loop, fut, self.epoch))
        return fut

    def reset(self) -> None:
        with self.lock:
            self.epoch += 1
            self.text = ""
            self.backlog = 0
            self.stats = {"chunks": 0, "busy_s": 0.0, "max_backlog": 0, "last_ms": 0.0}
            # 完整 PCM 已交离线复核，不再逐块计算旧队列。在途推理不强停。
            while True:
                try:
                    cmd, arg, fut, _ = self.q.get_nowait()
                except queue.Empty:
                    break
                if cmd == "flush":
                    self._resolve(arg, fut, "")
            self.q.put(("reset", None, None, self.epoch))

    @staticmethod
    def _resolve(loop, future, value):
        try:
            loop.call_soon_threadsafe(
                lambda: (not future.done()) and future.set_result(value))
        except RuntimeError:  # 页面/事件循环已经关闭
            pass

    def report(self) -> str:
        s = self.stats
        n = max(1, s["chunks"])
        return (f"{s['chunks']} 块 · 均 {s['busy_s']/n*1000:.0f}ms/块 · "
                f"积压峰值 {s['max_backlog']/16000*1000:.0f}ms")

    def _run(self):
        while True:
            cmd, arg, fut, epoch = self.q.get()
            try:
                with self.lock:
                    obsolete = epoch != self.epoch
                if obsolete:
                    if cmd == "flush":
                        self._resolve(arg, fut, "")
                    continue
                if cmd == "feed":
                    t0 = time.perf_counter()
                    txt = self.asr.feed(arg)
                    dt = time.perf_counter() - t0
                    with self.lock:
                        if epoch != self.epoch:
                            continue
                        self.text = txt
                        self.backlog = max(0, self.backlog - len(arg))
                        if dt > 0.005:                 # 攒块没推的那些帧不算一块
                            self.stats["chunks"] += 1
                            self.stats["busy_s"] += dt
                            self.stats["last_ms"] = dt * 1000
                elif cmd == "flush":
                    fl = getattr(self.asr, "flush", None)
                    txt = fl() if fl is not None else self.text
                    with self.lock:
                        if epoch == self.epoch:
                            self.text = txt or self.text
                        out = self.text if epoch == self.epoch else ""
                    self._resolve(arg, fut, out)
                elif cmd == "reset":
                    self.asr.reset()
            except Exception as e:                    # noqa: BLE001 识别挂了别把线程带走
                print(f"[asr] 线程里出错（{type(e).__name__}: {e}）", flush=True)
                if cmd == "flush":
                    with self.lock:
                        out = self.text if epoch == self.epoch else ""
                    self._resolve(arg, fut, out)

#: 纯声音的一轮拿什么文本入库。
#:
#: 这一轮一个字都没转出来，直接 ingest("") 抽不出任何事实、也就没有记忆行，之后
#: 问「刚才那首歌帮我重播」什么都找不到。给它一句话当载体。
#:
#: 但**必须是这个常量**，别在调用方各写各的字面量：核心靠它认出"这一轮其实没有
#: 说话"（见 orchestrator 里的 _sound_only），认不出来就不会打 sound_only 标签，
#: 回放挑候选时分不清"用户说话那轮"和"音乐那轮"，放出来是用户自己的声音。
SOUND_ONLY_TEXT = "用户放了一段声音给我听。"

#: 打印每一轮是「语义判完」还是「等满兜底」结束的，以及当时的分数。
#: 这两条路的延迟差好几百毫秒，不打日志根本分不清 EOT 到底有没有起作用。
EOT_DEBUG = os.environ.get("VOICEMEM_EOT_DEBUG", "0") != "0"
#: 打印「流式转写 → 离线复核」的前后对比。
REFINE_DEBUG = os.environ.get("VOICEMEM_REFINE_DEBUG", "0") != "0"
#: 分数要连续高过这个值、持续这么久，才认。见 feed() 里那段。
EOT_HOT_LEVEL = float(os.environ.get("VOICEMEM_EOT_HOT_LEVEL", "0.85"))
#: 文本侧的收尾信号：句末标点，或中文的句末语气词。
#: **为什么要有它**：Smart-Turn 是纯声学的，对升调疑问句判不准——实测
#: "我的饮食禁忌是什么？" 只有 0.38，而同一批陈述句是 0.97。可这种句子恰恰是
#: 对话里最常见的一类，靠声学永远等不到高分，只能干等 confirm_s 兜底。
#: 文本这边信号很干净：问号、句号、或"吗/呢/吧"结尾就是说完了。
_TEXT_DONE = re.compile(r"[。！？!?…]\s*$|[吗呢吧嘛呗]\s*[。！？!?]?\s*$")
#: 文本说"完了"时，声学分数至少要有这么高才认——低于它说明声学明确判定
#: "还在说"（句子中间的停顿常在 0.1 以下），那就别拿文本去推翻它。
EOT_TEXT_MIN = float(os.environ.get("VOICEMEM_EOT_TEXT_MIN", "0.3"))
EOT_HOT_S = float(os.environ.get("VOICEMEM_EOT_HOT_S", "0.15"))
_UNSET = object()


class VoiceStream:
    """核心流式输入会话：边喂边投机，说完时交出 Turn。

    ``vm.stream(on_partial=None, spec_min_chars=6, gamble_s=0.2, confirm_s=0.3,
    src_rate=24000)``。``feed`` 走内置流式 ASR + silero VAD；``feed_partial``
    接外部 ASR 的文本（换 ASR 只改喂进来的一行）。投机预取逻辑两者共用。

    两个时间参数是配套的，别单独调：

        静音 0ms ───────── 200ms ───────── 300ms
                            │               │
                     gamble_s：赌你说完了， confirm_s：VAD 确认说完，
                     后台开始检索记忆        记忆已经现成，直接开口

    中间那 100ms 就是留给检索的窗口——等 VAD 确认才开始查的话，这段时间会
    加在用户说完之后，变成他听得见的停顿。赌错了（用户只是停顿一下继续说）
    也不亏：下一帧有人声就把这次投机取消，白跑一次本地检索而已。
    """

    def __init__(self, vm, *, on_partial=None, spec_min_chars=6,
                 gamble_s=0.2, confirm_s=0.3, src_rate=24000, vad_threshold=None,
                 emotion=None, gate=_gate_mod.route, eot=None, textless_confirm_s=None,
                 turn_end_guard=None,
                 eot_min_s=float(os.environ.get("VOICEMEM_EOT_MIN_SILENCE", "0")),
                 eot_ends_turn=os.environ.get("VOICEMEM_EOT_ENDS_TURN", "1") != "0"):
        self.vm = vm
        #: 语义判「说完了没」（见 voicemem/utils/audio/eot.py）。``None`` = 不用，
        #: 回合仍按 confirm_s 掐表判——原来的行为。
        #:
        #: 用上它之后 confirm_s 的角色变了：它不再是"多久算说完"，而是"EOT 都没
        #: 表态时的兜底上限"。所以可以放宽（比如 0.8s）而不牺牲速度——真正决定
        #: 何时结束的是 EOT，掐表只在它拿不准时兜底。
        self.eot = eot
        #: 静音至少这么久才问 EOT。**默认 0 = 不要求静音**，纯看语义。
        #:
        #: 代价要知道：只看语义会在**句子中途的完整点**切断。实测一句 8.4 秒的话，
        #: 说到 5.5s 时 EOT 已经 0.87（"...what I'm going to do next week" 本身
        #: 就是完整的），而他接着说了 "which is my schedule"——0 的话就在那儿切了。
        #: 换来的是"说完立刻结束"，一点都不等。
        #:
        #: 觉得被切断太多就设 ``VOICEMEM_EOT_MIN_SILENCE=0.1``：要求语气落下来
        #: 100ms 才算数，能挡掉大部分句中误判，代价是慢 100ms。
        self.eot_min_s = eot_min_s
        #: EOT 能不能直接结束回合。**默认能**：语音一停就问它，它说齐了立刻结束，
        #: 掐表（``confirm_s``）退成"它一直不表态时的兜底上限"。
        #:
        #: 下面这段是当初判定它不可靠的记录，留着——但那批数据是在**缓冲区任意
        #: 长度**上打的分，取景每次都不一样（见 eot.py 的 ``_frame``）。裁成固定
        #: 取景之后重新量 1140 段：陈述句 0.97、疑问句 0.38~0.48，中间几乎没有样本，
        #: 是可以拿来判结束的——门槛得压到 0.5，0.6 正好从那个空档穿过去。
        #:
        #: 这不是保守，是实测：拿真实录音打分，一句完整的话说完时也只有 0.10~0.64
        #: （非母语英语、疑问句尾更低），拿它当结束判据要么切不断要么乱切。而同一
        #: 批数据里，句子中间稳定在 0.01~0.05——**相对高低是可信的，绝对值不可信**。
        #: 所以它适合做"要不要赌一把"，不适合做"这一轮结束了"。
        self.eot_ends_turn = eot_ends_turn
        #: 轮次闸门：``text -> "deep" | "shallow" | "backchannel"``。
        #:
        #: 默认就是 ``voicemem/gate.py`` 的三路判定（附和/浅/深）。
        #: ``None`` = 关掉闸门，每轮都检索（原来的行为）；传自己的函数就换掉判定。
        self.gate = gate
        #: 投机检索时带给 Search 的情绪提示（调用方可随时改写这个属性）。
        #:
        #: **右脑没有它就基本是空转的。** 右脑的情感记录几乎全部只挂在「情绪」
        #: 锚点上（实测一个库里 126 条 heartnote、124 条的锚点是 emotion），
        #: 不给情绪就一条都匹配不上，右脑只剩每轮都一样的静态 profile——回复里
        #: 读不出"它记得我这件事"，就是这么来的。实测同一个问题：
        #:     emotion=None   → 右脑 4 条，全是 profile
        #:     emotion="难过" → 情感记录 2 条 + 性格观察 1 条 + profile 2 条
        #:
        #: 但**本轮**的情绪读不得：``StreamState.emotion`` 是惰性属性，取一次要
        #: 同步跑整套声学感知（实测 2.1s），投机预取那 0–300ms 的预算根本不够。
        #: 所以调用方该把**上一轮** ingest 返回的 ``affect`` 写进来——情绪本来就有
        #: 连续性，代价是 0ms。
        self.emotion = emotion
        self.on_partial = on_partial
        self.spec_min_chars = spec_min_chars
        self.gamble_s = gamble_s
        self.confirm_s = confirm_s
        # Dialogue can probe short voiced input before streaming ASR emits a word.
        # Opt-in: library/music-only callers retain the original sound-only path.
        self.textless_confirm_s = textless_confirm_s
        # Optional application turn-taking policy. Applies to both EOT and timeout.
        self.turn_end_guard = turn_end_guard
        self._held_final_text = ""
        self._vad_silence = self._voiced_s = 0.0
        self._textless_probed = False
        self.src_rate = src_rate
        self.vad_threshold = vad_threshold
        # ASR/VAD 懒加载：feed_text / feed_partial（外部 ASR）不碰音频模型。
        self._asr = None
        self._vad = None
        # 回合状态
        self._text = ""
        self._silence = 0.0
        self._spoke = False
        self._speech_end = 0.0
        self._spec = None
        self._spec_text = ""
        self._spec_started = 0.0            # 上次起投机的时刻（节流用）
        self._gate_pre = None               # 静音一开始就在线程里预判的闸门（见 feed）
        self._gate_pre_text = ""
        self._eot_wait = 0.0                # 距上次问 EOT 过了多久（节流用）
        self._eot_score = 0.0               # 最近一次 EOT 打分
        self._raw_text = ""                 # 复核前的流式转写
        self._eot_hot = 0.0                 # 分数连续高了多久
        self._final_asr = _UNSET            # 离线复核 ASR（懒加载）
        self._route = _gate_mod.DEEP    # 本轮闸门判的路（见 _gate）
        self._last_memory = None   # 最新算好的投机记忆（SearchResult）
        self._pcm = []             # 本轮音频（16k 单声道），供 StreamState 按需做感知
        self._pcm_len = 0          # 已攒样本数，超上限就丢最早的（见 _MAX_TURN_S）
        self._preroll = []         # 开口前的一小段，接在本轮开头（见 _PREROLL_SAMPLES）

    @property
    def asr(self):
        if self._asr is None:
            self._asr = self.vm.utils.get("asr"); self._asr.reset()
        return self._asr

    @property
    def asr_worker(self) -> "_AsrWorker":
        """流式 ASR 的串行线程（见 _AsrWorker）。识别不再在事件循环里同步跑。"""
        w = getattr(self, "_asr_w", None)
        if w is None:
            w = self._asr_w = _AsrWorker(self.asr)
        return w

    @property
    def vad(self):
        if self._vad is None:
            if self.vad_threshold is None:
                self._vad = self.vm.utils.get("vad")   # 可注入：VoiceMem(vad=...) / config 的 vad 段
            else:
                from voicemem.utils.audio.stream_io import make_vad
                self._vad = make_vad(threshold=self.vad_threshold)
        return self._vad

    # ── 轮次闸门：这一句要不要检索 ──────────────────────────────────────────
    #
    # 闸门在**检索之前**，不在注入之前。放注入前的话检索照跑了，省下的只是几行
    # prompt——而检索才是这条路上的活儿。
    #
    # 判定随文本增长反复做，不是一锤子买卖：前 6 个字判成浅、涨到 12 个字冒出
    # "我上次"翻成深，那时再起检索也来得及（投机本来就是重复起的）。反过来
    # 由深翻浅就把已起的那次 cancel 掉。真正定生死的是**说完那一刻**那次判——
    # 那时整句在手，实测深句漏检约 1.4%，而只看前 15 个字是 46%。
    def _gate_locked(self, text) -> str:
        with _EMBED_LOCK:
            return self._gate(text)

    def _gate(self, text) -> str:
        if self.gate is None:
            return _gate_mod.DEEP
        try:
            return self.gate(text)
        except Exception as e:
            # 闸门自己坏了，宁可多检索一次，也不能因此把记忆丢了。
            print(f"[gate] 判定失败（{type(e).__name__}: {e}）→ 按 deep 走", flush=True)
            return _gate_mod.DEEP

    # ── 投机预取（本地分类器 + 本地向量 Search，0 LLM/网络，放线程里跟读麦克风并发）──
    async def _speculate(self, text) -> Turn:
        t0 = time.perf_counter()

        def work():
            from voicemem.leftbrain.query_embedding import query_embedding_scope
            started = time.perf_counter()
            # 只锁模型推理，不能持 TORCH_LOCK 等整个 search：search 会等待
            # 右脑子线程，而右脑 encode 也要这把锁，跨线程 RLock 仍会死锁。
            # 闸门保持线程内执行；分类/检索的 encode 由 _LockedEncoder 串行保护。
            route = self._gate_locked(text)
            if route != _gate_mod.DEEP:
                return route, None, started, started, time.perf_counter()
            with query_embedding_scope():
                c = self.vm.classify(text)
                classified = time.perf_counter()
                result = self.vm.search(text, slots=c.slots, entities=c.entities,
                                        emotion=self.emotion or None)
            searched = time.perf_counter()
            return route, result, started, classified, searched

        route, result, started, classified, searched = await asyncio.to_thread(work)
        self._route = route
        if result is None:
            return Turn(text, empty_result(), raw_text=text, route=route)
        print(f"[speculate] {text[:24]!r} -> {len(result.hits)} hits  "
              f"{(time.perf_counter()-t0)*1000:.0f}ms", flush=True)
        if os.environ.get("VOICEMEM_SEARCH_DEBUG", "0") != "0":
            print(f"[search-detail] 线程排队 {(started-t0)*1000:.0f}ms"
                  f" · 分类 {(classified-started)*1000:.0f}ms"
                  f" · 检索 {(searched-classified)*1000:.0f}ms"
                  f" · 回主循环 {(time.perf_counter()-searched)*1000:.0f}ms"
                  f" · 内部 {getattr(result, 'timing', {})}", flush=True)
        return Turn(text, result, raw_text=text, route=_gate_mod.DEEP)

    def _kick(self, text):
        """文本够长且变化了就（重）起后台投机——闸门放行的话。"""
        if not (text and text != self._spec_text and len(text) >= self.spec_min_chars):
            return
        now = time.monotonic()
        if (self._spec_started and now - self._spec_started < SPEC_MIN_INTERVAL_S
                and len(text) - len(self._spec_text) < SPEC_MIN_GROWTH):
            return                          # 节流：刚跑过、字也没多几个
        # 闸门在 _speculate 的线程里判（浅句在那边直接返回空结果），这里不再同步算。
        if self._spec:
            self._spec.cancel()
        self._spec_text, self._spec_started = text, now
        self._spec = asyncio.create_task(self._speculate(text))

    def _ready_memory(self):
        """取最新算好的投机记忆（SearchResult）；没算好就保持上一份/None。"""
        if self._spec is not None and self._spec.done() and not self._spec.cancelled():
            try:
                self._last_memory = self._spec.result().result
            except Exception:
                pass
        return self._last_memory

    @property
    def final_asr(self):
        """离线复核 ASR，取不到就是 None（沿用流式那份文本）。"""
        if self._final_asr is _UNSET:
            try:
                self._final_asr = self.vm.utils.get("asr_final")
            except Exception as e:
                print(f"[asr] 离线复核不可用（{type(e).__name__}: {e}）", flush=True)
                self._final_asr = None
        return self._final_asr

    def _transcribe_final(self, pcm):
        """Worker returns text only: cancellation must not overwrite a newer turn."""
        if pcm is None or not len(pcm):
            return None
        started = time.monotonic()
        model = self.final_asr
        loaded = time.monotonic()
        if model is None:
            return None
        try:
            return model.transcribe(pcm)
        except Exception as e:
            print(f"[asr] 复核失败（{type(e).__name__}: {e}）→ 沿用流式转写", flush=True)
            return None
        finally:
            if ASR_DEBUG:
                print(f"[asr-final] 获取/加载 {(loaded-started)*1000:.0f}ms"
                      f" · 推理 {(time.monotonic()-loaded)*1000:.0f}ms", flush=True)

    def _apply_refined(self, text):
        if text and text != self._text:
            if REFINE_DEBUG:
                print(f"[asr] 复核：{self._text!r} → {text!r}", flush=True)
            self._raw_text = self._text
            self._text = text

    def _refine(self, pcm) -> None:
        """Synchronous compatibility helper; the audio loop uses _refine_async."""
        self._apply_refined(self._transcribe_final(pcm))

    async def _refine_async(self, pcm) -> None:
        text = await self._final_text_async(pcm)
        self._apply_refined(text)

    async def _final_text_async(self, pcm):
        """Only return text: competing tasks never mutate turn state."""
        if pcm is None or not len(pcm):
            return
        submitted = time.monotonic()

        def work():
            if ASR_DEBUG:
                print(f"[asr-final] 排队 {(time.monotonic()-submitted)*1000:.0f}ms", flush=True)
            return self._transcribe_final(pcm)

        return await asyncio.get_running_loop().run_in_executor(_FINAL_ASR_EXECUTOR, work)

    def refine_current_snapshot(self) -> "asyncio.Task[str]":
        """Freeze current audio and return a background final-ASR transcript task.

        The snapshot is immutable: audio and streaming text received after this
        call cannot change the returned transcript.  If final ASR is unavailable
        or produces no text, the frozen streaming transcript is returned.
        """
        fallback = self._text.strip()
        pcm = np.concatenate(self._pcm).copy() if self._pcm else None

        async def resolve() -> str:
            refined = await self._final_text_async(pcm)
            return (refined or fallback).strip()

        return asyncio.create_task(resolve())

    async def _finish_asr(self, pcm):
        """完整音频复核与流式收尾并行；复核有效就不等落后的 partial 队列。"""
        started = time.monotonic()
        final = asyncio.create_task(self._final_text_async(pcm))
        flush = asyncio.ensure_future(self.asr_worker.flush())
        try:
            # The streaming worker already has the last stable characters. Give
            # the offline refiner a short head start, but never make the user
            # wait on a slow final pass just to release the turn.
            text = None
            try:
                text = await asyncio.wait_for(final, timeout=0.08)
            except asyncio.TimeoutError:
                text = None
            if text and text.strip():
                if flush.done() and not flush.cancelled():
                    self._text = flush.result() or self._text
                    self._apply_refined(text)
                    source = "完整复核（流式已就绪）"
                else:
                    # raw_text 原用于与提前下注文本比覆盖率。这里的流式文本缺尾巴，
                    # 必须用完整复核文本作比较，不能让旧半句话的下注误判为覆盖100%。
                    self._text = self._raw_text = text
                    self.asr_worker.reset()
                    source = "完整复核（绕过流式积压）"
            else:
                self._text = (await flush) or self._text
                source = "流式收尾（离线复核超时或无文本）"
            if ASR_DEBUG:
                print(f"[asr-final] {source} · ASR收尾 {(time.monotonic()-started)*1000:.0f}ms",
                      flush=True)
        finally:
            for task in (final, flush):
                if not task.done():
                    task.cancel()
            await asyncio.gather(final, flush, return_exceptions=True)

    async def _confirm(self) -> Turn:
        """说完那一刻：拿整句再判一次，这次的判定说了算。

        整句在手是这一步和前面每一次判的区别，也是准确率的来源——判别信息大量落在
        句子中后段（实测：只看前 6 字，深句漏检 84%；前 15 字 55%；整句 20%）。
        """
        route, pre = None, self._gate_pre
        self._gate_pre = None
        if pre is not None and len(self._text) - len(self._gate_pre_text) < SPEC_MIN_GROWTH:
            try:
                route = await pre                  # 多半已经算完了
            except Exception:
                route = None
        if route is None:
            route = await asyncio.to_thread(self._gate_locked, self._text)
        self._route = route
        if self._route != _gate_mod.DEEP:
            if self._spec:
                self._spec.cancel()
                self._spec = None
            print(f"[gate] {self._route}：{self._text[:16]!r} → 不检索", flush=True)
            return Turn(self._text, empty_result(), raw_text=self._raw_text or self._text,
                        route=self._route)
        try:
            # 前面几次判成浅、这次翻成深：那时没起检索，现在补一次。晚了一点，
            # 但整句的判定比前缀准得多，宁可晚也不能不查。
            turn = await (self._spec or self._speculate(self._text))
        except asyncio.CancelledError:
            turn = await self._speculate(self._text)
        # flush() 可能补出投机时还没解码出来的尾字：文本以最终版为准，记忆沿用已预取
        # 的结果（差的是最后几个字，为它重跑一次 Search 就把投机的收益还回去了）。
        if turn.text != self._text:
            turn = Turn(self._text, turn.result, raw_text=self._raw_text or self._text,
                        route=self._route)
        return turn

    def _reset_turn(self):
        if getattr(self, "_asr_w", None) is not None:
            self._asr_w.reset()
        elif self._asr is not None:
            self._asr.reset()
        self._text, self._silence, self._spoke = "", 0.0, False
        self._vad_silence = self._voiced_s = 0.0
        self._textless_probed = False
        self._speech_end = 0.0
        self._held_final_text = ""
        self._spec, self._spec_text, self._last_memory = None, "", None
        self._spec_started, self._gate_pre, self._gate_pre_text = 0.0, None, ""
        self._route = _gate_mod.DEEP
        self._eot_wait = 0.0
        self._eot_score = 0.0
        self._raw_text = ""
        self._eot_hot = 0.0
        self._pcm, self._pcm_len = [], 0
        self._preroll = []

    async def feed_text(self, text) -> Turn:
        """打字轮：闸门放行才检索。"""
        self._route = self._gate(text)
        if self._route != _gate_mod.DEEP:
            return Turn(text, empty_result(), route=self._route)
        return await self._speculate(text)

    async def feed_partial(self, text, ended: bool = False) -> StreamState:
        """接【外部 ASR】的 partial 文本（FunASR / Whisper / 任意流式 ASR）。

        换 ASR 只改「喂进来的这行文本」，本方法一字不用改。text 有新内容 = ``<speak>``
        并（重）起投机；``ended=True``（外部 VAD 判一句说完）→ 交出 Turn。
        """
        text = (text or "").strip()
        new = bool(text) and text != self._text
        if text:
            self._text = text
        if new and self.on_partial:
            self.on_partial(self._text)
        self._kick(self._text)
        if ended and self._text:
            turn = await self._confirm()
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm)
        return StreamState("<speak>" if new else "<silence>", self._text,
                           self._ready_memory(), None, self.vm,
                           silence=self._silence, spoke=self._spoke)

    async def feed(self, pcm_bytes) -> StreamState:
        """喂一块 PCM16（``src_rate``，默认 24k）：内置流式 ASR + silero VAD + 投机。
        每块返回 ``StreamState``（``<speak>``/``<silence>`` + 当前投机记忆 + 说完时的 Turn）。
        """
        received_at = time.monotonic()
        frame = resample(np.frombuffer(pcm_bytes, np.int16).astype(np.float32) / 32768.0,
                         src=self.src_rate)
        self._text = self.asr_worker.push(frame)     # 入队即返回，识别在 asr-worker 线程
        speaking = self.vad.is_speech(frame)
        if speaking:
            self._held_final_text = ""
        elif self._held_final_text:
            self._text = self._held_final_text
        if speaking:
            self._speech_end = received_at
            self._voiced_s += len(frame) / 16000.0
            self._vad_silence = 0.0
            self._textless_probed = False
        else:
            self._vad_silence += len(frame) / 16000.0

        # 攒本轮音频：StreamState 的感知字段按需取用（声纹/情绪），也用来存档回放。
        #
        # 原来是**每一帧都攒**，包括你没说话的那些。于是两轮之间的静音一直往里堆，
        # 堆到 30 秒上限，一句「早上好呀」存出来是 29.9 秒、RMS 0.005 的音频。
        # 情绪模型拿到这个只会判「低能量 = 难过」——实测 emotion2vec / SenseVoice /
        # 韵律三个模型在这种音频上全判难过，看着像模型不准，其实是喂错了东西。
        #
        # 现在只从**开口那一刻**开始录，前面留 PREROLL 秒不切掉字头。
        if speaking or self._spoke:
            if not self._spoke and self._preroll:      # 刚开口：把前摇接上
                self._pcm.extend(self._preroll)
                self._pcm_len += sum(len(f) for f in self._preroll)
                self._preroll = []
            self._pcm.append(frame)
            self._pcm_len += len(frame)
            while self._pcm_len > _MAX_TURN_SAMPLES and len(self._pcm) > 1:
                self._pcm_len -= len(self._pcm.pop(0))
        else:
            self._preroll.append(frame)                # 还没开口：只留最近一小段
            while sum(len(f) for f in self._preroll) > _PREROLL_SAMPLES and len(self._preroll) > 1:
                self._preroll.pop(0)
        # 还一个字都没转出来时，"有声音"就不算静音——那多半是音乐，而 VAD 只认
        # 人声（见 SOUND_LEVEL）。已经有转写了就老老实实按 VAD 来，别让环境噪音
        # 把一句话的结束拖住。
        audible = speaking or (
            not self._text.strip() and self._spoke
            and float(np.sqrt(np.mean(frame * frame))) >= SOUND_LEVEL)
        if audible:
            if speaking and self._silence > 0 and self._spec:   # barge-in：又开口了 → 丢弃这次投机
                self._spec.cancel(); self._spec, self._spec_text = None, ""
            if speaking:
                self._spoke = True
                self._gate_pre = None              # 又开口了：静音时预判的闸门作废
            self._silence = 0.0
        else:
            if self._silence == 0.0 and self._spoke and self._text.strip():
                # 刚静下来：闸门判定（要过一遍 embedding，200ms 上下）先起在线程里，
                # 跟 200ms 静音确认**并行**。原来是确认完才判，那 200ms 白白串在
                # "闭嘴→回复开始"里。判的是流式文本；复核后差几个字不影响深浅。
                self._gate_pre_text = self._text
                self._gate_pre = asyncio.ensure_future(asyncio.to_thread(self._gate_locked, self._text))
            self._silence += len(frame) / 16000.0
        if self._text.strip() and self.on_partial:
            self.on_partial(self._text)
        # 边说边预取 / 200ms 赌说完补发
        if self._spoke and self._text.strip() and \
                (self._silence == 0.0 and len(self._text) >= self.spec_min_chars
                 or self._silence >= self.gamble_s):
            self._kick(self._text)
        # 一轮成立要有转写文本——否则每段环境噪声都会变成一轮。
        # 但有个例外：**对着麦克风放音乐**。VAD 会把音乐判成人声（实测 357 块全
        # 中），ASR 却一个字都转不出，于是永远不成一轮：没有记忆、没有存档音频，
        # 之后问「刚才那首歌帮我重播」当然找不到。放够久（≥ MIN_SOUND_ONLY_S）
        # 就当作一轮交出去，text 为空，由上层决定怎么记。
        sound_only = (not self._text.strip()
                      and self._pcm_len >= MIN_SOUND_ONLY_S * 16000)
        # 一个字都没转出来的这一轮多半是音乐，别拿对话的 300ms 去切它，
        # 见 SOUND_ONLY_SILENCE_S。
        need_silence = self.confirm_s if self._text.strip() else SOUND_ONLY_SILENCE_S
        # 语义判完：静音过了 eot_min_s（100ms）就开始问 EOT，它说齐了立刻结束，
        # 不用等满 confirm_s。两个条件都要，理由见 __init__ 里 eot_min_s 那段。
        semantic_done = False
        # 说话期间也打分（供上层提前起跑生成用），静音期间的那次才参与回合判定。
        if self.eot is not None and self._spoke and self._text.strip() and self._pcm:
            self._eot_wait += len(frame) / 16000.0
            if self._eot_wait >= 0.04:          # 15ms 一次，每帧都问是浪费
                self._eot_wait = 0.0
                try:
                    _p = self.eot.score(np.concatenate(self._pcm))
                    # **要连续高才算**，单帧不作数。实测句子中间会冒孤立的假高峰
                    # （英文 1.4s 处 0.75、中文 0.6s 处 0.97，前后都是 0.01），
                    # 而真正说完时是连着一串 0.9+。只看单帧就会在两个词之后下注，
                    # 那份回复只听到 "I want"，必然作废——等于赌了个必输的。
                    self._eot_hot = (self._eot_hot + 0.04) if _p >= EOT_HOT_LEVEL else 0.0
                    # 两条路认一条：声学连续高（陈述句走这条），或者文本已经收了尾
                    # 而声学也没在说"还没完"（疑问句走这条）。
                    _text_done = bool(_TEXT_DONE.search(self._text or ""))
                    self._eot_score = _p if (
                        self._eot_hot >= EOT_HOT_S
                        or (_text_done and _p >= EOT_TEXT_MIN)) else 0.0
                    # 只有「静音够久」那次才算数——只看分数会在句子中途的语义完整点
                    # 切断（实测一句 8.4s 的话，5.5s 处就已经 0.87）。
                    # **语音一停就问，问到就切**，不再等掐表。拿 1140 段真实回合
                    # 录音量过：同一段话在静音 0ms 和 200ms 处问，EOT 给的分数中位
                    # 都是 0.85——分数在语音结束那一刻就已经到位了，等那 200ms 纯粹
                    # 是白等（`_frame` 本来就把尾部裁到固定 200ms 取景，缓冲区里
                    # 多出来的静音根本进不了模型）。
                    #
                    # 唯一的硬要求是 `_silence > 0`：至少要有一帧不在说话。说话中
                    # 途（silence==0）也判的话会在句子中间的语义完整点切断，而且会
                    # 把附和的窗口（停顿 100~280ms）整个吃掉——附和和回合结束抢的是
                    # 同一个停顿，得让 EOT 来分：它说齐了就结束回合，它说没齐，这个
                    # 停顿才轮到附和"嗯"一声。
                    semantic_done = (self.eot_ends_turn and _p >= self.eot.threshold
                                     and self._silence > 0
                                     and self._silence >= self.eot_min_s)
                    if EOT_DEBUG and self._silence > 0:
                        print(f"[eot] 静音 {self._silence*1000:3.0f}ms  分数 {_p:.2f}"
                              f"  {'→ 结束' if semantic_done else ''}", flush=True)
                except Exception as e:
                    print(f"[eot] 判定失败（{type(e).__name__}: {e}）→ 退回掐表",
                          flush=True)
                    self.eot = None
        # "ok" often has no streaming hypothesis at all. Do not classify that
        # as music and wait for 5s of audio + 3s silence. Use VAD silence (not
        # residual speaker energy) for one offline probe per voiced burst.
        fast_final = False
        if (self.textless_confirm_s is not None and self._spoke
                and not self._text.strip() and not self._textless_probed
                and self._voiced_s >= 0.12
                and self._vad_silence >= max(self.confirm_s, self.textless_confirm_s)):
            self._textless_probed = True
            _asr_started = time.monotonic()
            pcm = np.concatenate(self._pcm) if self._pcm else None
            refined = await self._final_text_async(pcm)
            if refined and refined.strip():
                self._text = self._raw_text = refined.strip()
                self.asr_worker.reset()
                fast_final = True
                if ASR_DEBUG:
                    print(f"[asr-final] 无流式字短句快速复核 · VAD静音 {self._vad_silence*1000:.0f}ms", flush=True)
            # Empty/failed recognition is not a user turn; keep the sound-only
            # buffer intact and do not repeatedly probe the same silence.
        allow_end = (self.turn_end_guard is None or self.turn_end_guard(
            self._text, self._silence, speaking, len(frame) / 16000.0,
            float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0))
        if fast_final and not allow_end:
            self._held_final_text = self._text
        if allow_end and (fast_final or (self._spoke and (semantic_done or self._silence >= need_silence)
                          and (self._text.strip() or sound_only))):
            if EOT_DEBUG and self.eot is not None:
                print(f"[eot] 回合结束：{'短句快速复核' if fast_final else '语义判定' if semantic_done else f'兜底掐表 {need_silence*1000:.0f}ms'}"
                      f"（静音 {self._silence*1000:.0f}ms）", flush=True)
            if not fast_final:
                _asr_started = time.monotonic()
            if ASR_DEBUG:
                print(f"[asr] 本轮 {self.asr_worker.report()}", flush=True)
            pcm = np.concatenate(self._pcm) if self._pcm else None
            # 说完了：用离线模型把整轮**重转一遍**，这份才进记忆、才给回复模型。
            # 流式那份是为低延迟牺牲了准确率的，只配用来判打断/EOT/闸门。
            # 见 utils/audio/asr.py 的 OfflineASR：实测同一批录音，流式把
            # "Everything is good." 转成 "Everything is going"，离线一字不差，37ms。
            if not fast_final and not self._held_final_text:
                await self._finish_asr(pcm)
                # Final ASR may reveal a dangling clause absent from the partial.
                # Retain that decode while the app leaves room for continuation.
                if self.turn_end_guard is not None and not self.turn_end_guard(
                        self._text, self._silence, speaking, 0.0,
                        float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0):
                    self._held_final_text = self._text
                    return StreamState("<silence>", self._text, self._ready_memory(),
                                       None, self.vm, silence=self._silence,
                                       spoke=self._spoke, eot_score=0.0,
                                       speech_end=self._speech_end)
            _refined = time.monotonic()
            turn = await self._confirm()                   # VAD 确认说完 → 交出预算记忆
            if ASR_DEBUG:
                print(f"[asr-final] 判停/VAD后等待 {(_asr_started-self._speech_end)*1000:.0f}ms"
                      f" · ASR并行收尾 {(_refined-_asr_started)*1000:.0f}ms"
                      f" · 回合确认 {(time.monotonic()-_refined)*1000:.0f}ms", flush=True)
            speech_end = self._speech_end
            self._reset_turn()
            return StreamState("turn_over", turn.text, None, turn, self.vm, pcm,
                               speech_end=speech_end)
        return StreamState("<speak>" if speaking else "<silence>", self._text,
                           self._ready_memory(), None, self.vm,
                           silence=self._silence, spoke=self._spoke,
                           eot_score=self._eot_score, speech_end=self._speech_end)
