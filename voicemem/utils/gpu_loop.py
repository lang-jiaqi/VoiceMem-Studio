"""单条 GPU 流:所有 MLX GPU 活儿都排一个线程提交,避免驱动崩。

**为什么要有这个东西。** Apple 的 AGX/IOGPU 驱动有个并发 bug:两个线程各自往
同一块 GPU 提交命令缓冲时,内部那个"借内存 +1 / 还内存 -1"的计数不是线程安全的,
会下溢成负数,驱动一看账坏了直接 kernel panic,看门狗把整台机器重启
(``completeMemory() prepare count underflow`` @IOGPUMemory.cpp)。实测本地同时跑
Qwen-TTS 和本地 LLM,一天崩三次。

而 MLX 的 GPU 流是**线程本地**的:在别的线程里用另一个线程建的流,直接
``RuntimeError: There is no Stream(gpu, 0) in current thread``。所以没法"共享一条
流对象",只能**共享一个拥有流的线程**——所有 GPU 提交都发生在它一个人身上,
驱动永远只看到一个提交者,那个计数就不会乱。

**为什么不是简单串行。** 语音要边生成边合成:LLM 还在生成后半句时,TTS 已经在
念前半句了。如果 LLM 整段跑完再交给 TTS,用户要干等。所以这个线程做**轮询**:
每个活跃的生成器走一步(LLM 出一个 token、TTS 出一块音频),再轮到下一个。单线程
不并发,但两边都在推进——这正是别人不崩、又不牺牲流水的做法。

    from voicemem.utils.gpu_loop import gpu_loop
    # 流式:把"造一个生成器"的函数交进去,逐项取回
    for kind, item in gpu_loop().iter(lambda: my_mlx_generator()):
        ...
    # 一次性:在 GPU 线程上跑完一个函数,拿返回值
    val = gpu_loop().call(lambda: heavy_mlx_thing())
"""
from __future__ import annotations

import queue
import threading

#: 每项取回来的形状:("ok", 值) 正常,("err", 异常) 出错,然后一个 None 收尾。
_DONE = None


class Job:
    """一次提交。``out`` 是取结果的队列,``cancel()`` 让 GPU 线程别再推进它。

    **取消必须存在**:提前生成(EOT 赌注)赌错时上层只是"不要这份结果"了,可生成器
    还在 GPU 线程上一个 token 一个 token 地跑,最多跑满 512 个才停——真正那一轮就
    排在它后面干等。实测赌错的那轮，真回合"排队等 GPU 线程"要 400~1300ms。
    """

    __slots__ = ("out", "_cancelled")

    def __init__(self):
        self.out: queue.Queue = queue.Queue()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled


class GpuLoop:
    """独占一条 GPU 流的线程,轮询驱动所有交进来的生成器。"""

    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()   # 新任务:(make_gen, out_queue)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _ensure(self):
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="gpu-loop", daemon=True)
                self._thread.start()

    def on_thread(self) -> bool:
        """当前就在 GPU 线程上吗。用来避免自己给自己派活儿:已经在这条线程里的
        代码(比如生成器内部要加载模型)直接干就行,再 ``call`` 一次会死等自己。"""
        return threading.current_thread() is self._thread

    def iter(self, make_gen, weight: int = 1) -> "Job":
        """把 ``make_gen``(一个**在 GPU 线程上**被调用、返回生成器的函数)排进来。

        返回一个队列:每步吐 ``("ok", 项)`` 或 ``("err", 异常)``,结束吐 ``None``。
        ``make_gen`` 故意延到 GPU 线程里才调——生成器第一次 ``next`` 往往就建模型、
        做 prefill,这些都必须发生在拥有流的那个线程。

        ``weight``:每轮连着走几步。轮询是公平的 1:1,但 TTS 要跟上播放、LLM 只要
        喂饱分句——给 TTS 高一点(比如 4),它就不会因为跟 LLM 平摊算力而欠载卡顿。
        """
        job = Job()
        self._jobs.put((make_gen, job, max(1, weight)))
        self._ensure()
        return job

    def call(self, fn):
        """在 GPU 线程上把 ``fn`` 跑完,返回它的返回值(或抛它抛的异常)。

        一次性的活儿(加载模型、预热 KV)用它。内部就是"只 yield 一次的生成器"。
        """
        job = self.iter(lambda: iter((fn(),)))
        tag, payload = job.out.get()      # 只有一项
        job.out.get()                     # 把收尾的 None 取掉
        if tag == "err":
            raise payload
        return payload

    def _run(self):
        import mlx.core as mx
        mx.set_default_device(mx.gpu)     # 这个线程从此拥有 GPU 流
        active: list[list] = []           # [[生成器, out_queue, weight], ...]
        while True:
            # 先收新任务:没有活跃的就阻塞等,有的话只非阻塞捞一遍,别耽误轮询。
            if not active:
                job = self._jobs.get()
                self._begin(job, active)
            else:
                try:
                    while True:
                        self._begin(self._jobs.get_nowait(), active)
                except queue.Empty:
                    pass
            # 每个活跃生成器走 weight 步再轮到下一个:TTS 权重高,跟得上播放;
            # LLM 权重 1,喂饱分句就行。全在这一条线程,从不并发提交。
            still: list[list] = []
            for entry in active:
                gen, job, weight = entry
                out = job.out
                if job.cancelled:              # 上层不要了 → 立刻收手，别再占 GPU
                    gen.close()
                    out.put(_DONE)
                    continue
                alive = True
                for _ in range(weight):
                    try:
                        item = next(gen)
                    except StopIteration:
                        out.put(_DONE)
                        alive = False
                        break
                    except Exception as e:           # noqa: BLE001 交给调用方决定
                        out.put(("err", e))
                        out.put(_DONE)
                        alive = False
                        break
                    out.put(("ok", item))
                if alive:
                    still.append(entry)
            active = still

    @staticmethod
    def _begin(pending, active):
        make_gen, job, weight = pending
        try:
            active.append([make_gen(), job, weight])
        except Exception as e:                        # noqa: BLE001
            job.out.put(("err", e))
            job.out.put(_DONE)


_LOOP: GpuLoop | None = None
_LOOP_LOCK = threading.Lock()


def gpu_loop() -> GpuLoop:
    """进程里唯一的那条 GPU 流。第一次用时才建线程。"""
    global _LOOP
    if _LOOP is None:
        with _LOOP_LOCK:
            if _LOOP is None:
                _LOOP = GpuLoop()
    return _LOOP
