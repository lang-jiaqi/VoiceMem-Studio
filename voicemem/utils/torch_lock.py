"""进程里唯一的 torch 锁。

torch 的 MPS 后端不能两条线程同时提交：流式 ASR（自己的线程）、闸门/检索/入库的
embedding（线程池）、场景分类（后台）一撞就是 Metal 断言
"A command encoder is already encoding to this command buffer"，进程直接 abort。
所有会碰 torch 模型的调用都包一层 ``with TORCH_LOCK``，同一时刻只有一个在算。
RLock：闸门里套 embedding 这种嵌套调用不会自己把自己锁死。
只允许包住本线程的模型计算；禁止持锁等待子线程/future 或包住整个 Search。
RLock 只能让同一线程重入，不能让右脑子线程重入父检索线程持有的锁。
MLX（LLM / Breeze）有自己的单线程 gpu_loop，不走这把锁。
"""
import threading

TORCH_LOCK = threading.RLock()
