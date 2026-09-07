"""本地回复模型（MLX，Apple silicon）：把不变的那截 prompt 一次算好，一直留着。

本地跑 LLM 本身**更慢**——prefill 是纯算力活（工作量 ≈ 2 × 参数量 × token 数），
这台机器实测 2.4ms/token，700 token 的 prompt 全量算要 1.9 秒。

但本地有云端给不了的东西：**KV cache 我们自己控**。prompt 长这样：

    [system 人设 ~630 token]  [历史 6 轮]  [user: 记忆+提示 + 这句话]
     └──── 一场都不变 ────┘   └ 末尾追加 ┘  └───── 每轮全新 ─────┘

前面那截每轮都一样，却曾经每轮都在重算——因为 ``stream_generate`` 会把这一轮的
尾巴和生成出来的 token 都追加进 cache，用完就不是那个前缀了。现在改成**母本 +
复制**：母本只存前缀、永不参与生成，每轮复制一份出来用（``_fork``，几毫秒）。

实测（4B-4bit，320 token 前缀）：

    从头算            首字 1260 ms
    复制母本再生成      首字  280 ms   ← 输出逐字相同

母本在开服时就建好（见 web/run.py 的启动预热），第一个用户不用替它垫这一秒半。
之后每轮只算「历史增量 + 记忆 + 这句话」。

OpenAI 那边做不到这件事：它的 prompt 缓存要求**可复用前缀本身** ≥1024 token
（我们人设才 630），而且过期时间不由我们定。

代价说清楚：
  · 母本占显存，约 52MB 固定 + 33KB/token
  · 4B 的回复质量比 gpt-4o 低一档
  · 首次加载模型要几十秒

    from voicemem.local_llm import LocalLLM
    vm = VoiceMem(reply=LocalLLM())
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from voicemem.utils.gpu_loop import gpu_loop

#: 默认模型。4B 是实测下来的平衡点：9B 慢一倍（3060ms），更小的质量不够。
MODEL = os.environ.get("VOICEMEM_LOCAL_LLM", "mlx-community/Qwen3.5-4B-4bit")
#: 打印每轮的缓存命中和等锁时间。没有它就只能猜为什么慢。
DEBUG = os.environ.get("VOICEMEM_LLM_DEBUG", "0") != "0"
#: 渲染前缀时占位用；只要不会出现在真实文本里就行。
_MARK = "\x00__VM__\x00"


class LocalLLM:
    """可直接传给 ``VoiceMem(reply=...)`` 的本地回复 provider。

    ``prewarm(history)`` 在用户开口时调（放后台线程），``__call__`` 在他说完时调。
    两者之间共享一份 prompt cache；``prewarm`` 没来得及跑完就退回全量，不会出错。
    """

    def __init__(self, model_name: str = MODEL, system: str | None = None):
        self.model_name = model_name
        # 人设按空间语言取（voicemem/persona.py）。这里定一次就不再变——
        # 语言是空间的属性，一个实例只服务一个空间。
        from voicemem.reply import default_system
        self.system = system or default_system()
        self._m = self._tok = None
        # 两级母本。**人设那一级永不失效**：它一个字都不会变，可对话历史每轮都在
        # 长、还会被误打断和换说话人冲乱。以前只有一级，历史一岔就把整份丢掉——
        # 实测日志里 "前缀在第 632/750 个 token 分叉"，而人设正好 631 个 token：
        # 匹配得好好的却跟着被重算，每轮白烧 1.5 秒。
        self._base = None                    # [人设]，开服时建，之后只读
        self._base_ids: list[int] = []
        self._cache = None                   # [人设][历史][记忆]，会被历史冲掉
        self._cached_prefix: list[int] = []

    def set_system(self, system: str) -> None:
        """换人设（比如切到另一种语言的空间）。变了就把两级母本都丢掉。

        人设是所有缓存前缀的开头：换了它，连"永不失效"的底座也不再是这次 prompt
        的前缀了。不丢就会拿旧人设的 KV 去生成——那不是慢，是答错。
        """
        if system and system != self.system:
            self.system = system
            self._base, self._base_ids = None, []
            self._cache, self._cached_prefix = None, []

    def load(self):
        """加载权重。**必须发生在 GPU 线程上**——MLX 的流是线程本地的,别的线程
        加载、这个线程用会直接崩(还有和 TTS 抢 GPU 的驱动 panic,见 gpu_loop)。
        不在 GPU 线程时就把加载派给它;已经在上面(生成器内部)就直接加载。"""
        if self._m is None:
            from mlx_lm import load
            from voicemem.detokenizer_cache import cache_bpe_vocabulary

            def load_cached():
                model, tokenizer = load(self.model_name)
                if cache_bpe_vocabulary(tokenizer) and DEBUG:
                    print("[llm] BPE 只读词表已缓存，每轮解码状态独立", flush=True)
                return model, tokenizer

            loop = gpu_loop()
            if loop.on_thread():
                self._m, self._tok = load_cached()
            else:
                self._m, self._tok = loop.call(load_cached)
        return self._m, self._tok

    def _render(self, msgs) -> str:
        _, tok = self.load()
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)

    def _msgs(self, text: str, memory_context: str, history):
        out = [{"role": "system", "content": self.system}] + list(history or [])
        out.append({"role": "user",
                    "content": f"{memory_context}\n\n{text}" if memory_context else text})
        return out

    def _fork(self, src_cache):
        """从母本"复制"一份可以拿去生成的 cache——**只换容器，不拷数组**。

        母本自己永远不参与生成：``stream_generate`` 会把这一轮的尾巴和生成出来的
        token 都追加进 cache，用完就不是那个前缀了。以前的做法是"一份缓存只用
        一次"，于是每轮都要把人设那 600 多个 token 重算一遍（1.7 秒）；而人设一个
        字都没变过。

        为什么可以不拷：两种 cache 追加时都是**分配新数组**，不会往原数组里写。
        ``KVCache`` 的 keys/values 被 state setter 设成正好等于前缀长度，下次追加
        必然先扩容；``ArraysCache`` 是把新数组赋进 list 的槽位，换了新 list 就动
        不到母本那个。真拷一份要 75MB，几轮叠起来直接 Metal OOM（实测第一次生成
        被换页拖到 13.6 秒）。

        state 的容器类型要原样保留：``ArraysCache``（线性注意力层，定长）给的是
        list，``KVCache`` 给的是 tuple，塞错类型 setter 会直接报错。
        """
        from mlx_lm.models.cache import make_prompt_cache
        model, _ = self.load()
        out = make_prompt_cache(model)
        for dst, src in zip(out, src_cache):
            st = src.state
            dst.state = (list(st) if isinstance(st, list)
                         else tuple(st) if isinstance(st, tuple) else st)
        return out

    def _prefix_ids(self, memory_context, history) -> list[int]:
        """[人设][历史][这轮记忆] 的 token——到用户那句话之前为止。

        用一个标记渲染再截断，而不是"末尾少留 32 个 token"。省下的几十个 token
        不是重点，重点是截断点得**确定**：只有确定，记忆到货时才能往同一份缓存
        上续算，而不是整份重来。
        """
        _, tok = self.load()
        full = self._render(self._msgs(_MARK, memory_context, history))
        ids = tok.encode(full[:full.index(_MARK)])
        # 末尾丢一个 token：head 单独分词时，最后一个 token 可能跟用户那句话的
        # 第一个字合并成别的 token，那就不是真前缀了。
        return ids[:-1]

    def prewarm(self, history=None, memory_context: str = "", *, cancelled=None) -> int:
        """把「人设 + 历史 + 记忆」过一遍模型，KV 存进**母本**。返回缓存了多少 token。

        母本只存前缀、永不参与生成（生成用 _fork 复制的容器）。开服时调一次把人设
        算进去，之后每轮说完再调一次把新增的那轮历史续上——续算只花多出来那一截
        的钱，几十毫秒。整件事派到 GPU 线程上做(见 gpu_loop),不然会跟 TTS 抢流。
        """
        # asyncio 取消 to_thread 不会停掉已经入队的 GPU 工作。必须在 GPU
        # 线程真正执行前再检查一次，避免过期预热挡住新回复；已开始的运算不强停。
        def run():
            if cancelled is not None and cancelled.is_set():
                return 0
            return self._prewarm(history, memory_context)

        if cancelled is not None and cancelled.is_set():
            return 0
        return gpu_loop().call(run)

    def _prewarm(self, history, memory_context) -> int:
        import mlx.core as mx
        model, _ = self.load()
        ids = self._prefix_ids(memory_context, history)
        if not ids:
            return 0
        if ids == self._cached_prefix and self._cache is not None:
            return len(ids)                          # 这一份已经热过了
        cache, done, kind = self._pick_base(ids)
        if done < len(ids):
            model(mx.array(ids[done:])[None], cache=cache)
            mx.eval([c.state for c in cache])
        self._cache, self._cached_prefix = cache, ids
        # 头一次(开服预热人设)顺手把它存成永久底座，以后历史怎么乱都还能用。
        if self._base is None:
            self._base, self._base_ids = cache, list(ids)
            kind = "人设底座 "
        if DEBUG:
            print(f"[llm] 母本{kind}{len(ids) - done} token → 共 {len(ids)}", flush=True)
        return len(ids)

    def _pick_base(self, ids):
        """挑一份能接着往下算的 cache：越长越好，接不上就退到人设底座。

        返回 ``(可写的 cache, 已经算好多少 token, 日志用的说法)``。挑到的那份要
        先 fork 再往上写——直接写会把母本本身撑长，下一轮就对不上了。
        """
        from mlx_lm.models.cache import make_prompt_cache
        for cache, pre, kind in ((self._cache, self._cached_prefix, "续 "),
                                 (self._base, self._base_ids, "接人设底座 ")):
            if cache is not None and len(pre) <= len(ids) and ids[:len(pre)] == pre:
                return self._fork(cache), len(pre), f"{kind}{len(pre)}+"
        return make_prompt_cache(self.load()[0]), 0, "建 "

    def drop_cache(self) -> None:
        """丢掉历史那一级。人设底座不动——它永远是对的。"""
        self._cache, self._cached_prefix = self._base, list(self._base_ids)

    def _gen_iter(self, text, memory_context, history, submitted=0.0):
        """流式吐字——**整个跑在 GPU 线程上**(见 gpu_loop)。渲染、分词、fork、
        forward 全在这一条流里,跟 TTS 轮流推进,从不并发提交,驱动就不会崩。

        每 ``next`` 出一个 token,交回 gpu_loop 后它就去推进 TTS 一步,再回来。
        """
        import time as _t
        from mlx_lm import stream_generate
        _t1 = _t.perf_counter()
        _queued = (_t1 - submitted) * 1000 if submitted else 0.0
        model, tok = self.load()
        ids = tok.encode(self._render(self._msgs(text, memory_context, history)))
        # 缓存只有在**确实是这次 prompt 的前缀**时才能用（逐 token 比对），
        # 对不上就退一级：历史那份岔了，人设那份还在，起码省下 600 多个 token。
        cache = prefix = None
        _tf = _t.perf_counter()
        for c, pre in ((self._cache, self._cached_prefix),
                       (self._base, self._base_ids)):
            if c is not None and len(pre) < len(ids) and ids[:len(pre)] == pre:
                cache, prefix = self._fork(c), pre
                break
        _fork_ms = (_t.perf_counter() - _tf) * 1000
        hit = cache is not None
        if not hit:
            prefix = []
        # 直接喂 token id，不要 decode 再让 stream_generate 重新分词：那一趟
        # 往返可能切出不一样的 token，前缀就白对了。
        tail = ids[len(prefix):] if hit else ids
        if DEBUG:
            print(f"[llm] prompt {len(ids)} token｜"
                  f"{'命中缓存 ' + str(len(prefix)) if hit else '没命中'}｜"
                  f"要算 {len(ids) - (len(prefix) if hit else 0)} token", flush=True)
            if not hit and prefix:
                i = next((k for k, (a, b) in enumerate(zip(ids, prefix)) if a != b),
                         min(len(ids), len(prefix)))
                print(f"[llm]   前缀在第 {i}/{len(prefix)} 个 token 分叉｜"
                      f"生成 {tok.decode(ids[i:i+24])!r}｜"
                      f"预热 {tok.decode(prefix[i:i+24])!r}", flush=True)
        # mlx-lm 的进度回调在 prompt 计算开始、每批 cache eval 后、首 token
        # eval 后触发。不要另加 mx.eval：那会改变我们正在测的执行方式。
        progress = {}

        def prompt_progress(done, total):
            now = _t.perf_counter()
            if done == 0:
                progress["start"] = progress["prefill"] = now
            elif done < total:
                progress["prefill"] = now
            else:
                progress["ready"] = now

        _g0 = _t.perf_counter()
        first_token = True
        first_text = True
        empty_tokens = 0
        try:
            for r in stream_generate(model, tok, tail, max_tokens=512,
                                     prompt_cache=cache if hit else None,
                                     **({"prompt_progress_callback": prompt_progress} if DEBUG else {})):
                now = _t.perf_counter()
                if DEBUG and first_token:
                    print(f"[llm] 首token {(now - _g0) * 1000:.0f}ms"
                          f"（排队等 GPU 线程 {_queued:.0f} / 渲染+分词 "
                          f"{(_g0 - _t1) * 1000 - _fork_ms:.0f} / fork {_fork_ms:.0f}）",
                          flush=True)
                    if "ready" in progress and "start" in progress:
                        print(f"[llm-detail] 生成器准备 {(progress['start']-_g0)*1000:.0f}ms"
                              f" · 尾部prefill {(progress['prefill']-progress['start'])*1000:.0f}ms"
                              f" · 首token求值/流水准备 {(progress['ready']-progress['prefill'])*1000:.0f}ms"
                              f" · 解码返回 {(now-progress['ready'])*1000:.0f}ms"
                              f" · 未缓存 {len(tail)} token", flush=True)
                first_token = False
                if first_text:
                    if r.text:
                        if DEBUG:
                            print(f"[llm-text] 首个可显示文字 {(now-_g0)*1000:.0f}ms"
                                  f" · 前置空文本token {empty_tokens}", flush=True)
                        first_text = False
                    else:
                        empty_tokens += 1
                yield r.text
        finally:
            # 这一轮的副本用完就还:不显式清,MLX 的缓冲池会一直攒着,几轮就 OOM。
            del cache
            try:
                import mlx.core as mx
                mx.clear_cache()
            except Exception:
                pass

    async def __call__(self, text: str, memory_context: str = "",
                       history: list | None = None) -> AsyncIterator[str]:
        """流式吐字。生成器跑在共享 GPU 线程上(gpu_loop),用普通队列把 delta 递
        回来;这里在事件循环侧把阻塞的队列 get 丢进 executor,不卡读麦克风那条线。"""
        loop = asyncio.get_running_loop()
        import time as _t
        _sub = _t.perf_counter()

        def stamped():
            gen = self._gen_iter(text, memory_context, history, _sub)
            try:
                for value in gen:
                    yield value, _t.perf_counter()
            finally:
                gen.close()  # 仍在 GPU 线程上释放被取消的生成器和 cache

        job = gpu_loop().iter(stamped)
        first_received = True
        try:
            while True:
                item = await loop.run_in_executor(None, job.out.get)
                if item is None:
                    return
                kind, val = item
                if kind == "err":
                    print(f"[llm] 本地生成失败：{type(val).__name__}: {val}", flush=True)
                    return
                val, produced_at = val
                if val:
                    if first_received and DEBUG:
                        now = _t.perf_counter()
                        print(f"[llm-delivery] 首文字回主循环 {(now-produced_at)*1000:.0f}ms"
                              f" · 提交到收到 {(now-_sub)*1000:.0f}ms", flush=True)
                    first_received = False
                    yield val
        finally:
            # 上层不要了（提前生成赌错被丢弃、或用户打断）就**立刻叫停 GPU 那边**。
            # 不叫停的话它会一直生成到 max_tokens，占着单条 GPU 流，真正那一轮排在
            # 后面干等——实测"排队等 GPU 线程"因此要 400~1300ms。
            job.cancel()
