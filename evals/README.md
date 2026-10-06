# 评测脚本

这里保留手动运行的延迟、检索效果和模型质量评测。自动回归测试与运行方法见 [tests/README.md](../tests/README.md)。

| 评测范围 | 脚本 |
| --- | --- |
| 对话和语音延迟 | `latency.py`、`deepseek_latency.py`、`breeze_cuda_latency.py` |
| 回合结束与打断 | `eot_turn_end.py`、`echo_barge.py` |
| 向量编码与检索 | `embed_latency.py`、`query_embedding_latency.py`、`embed_ab.py`、`embed_locomo.py` |
| 检索排序和相关性 | `topk_relevance.py`、`recency_ab.py`、`recency_rerank.py` |
| 路由质量 | `router_quality.py`、`turn_gate.py` |
| 标签维护对比 | `retag_llm.py` |

每个脚本文件顶部说明其参数和所需数据。它们可能需要模型、API、音频样本或专用 Memory Space，并非全部可以在干净环境中直接运行。不要使用个人正在使用的记忆库做实验；运行会写入数据的模式前，先阅读对应参数说明。

这里的评测用于观察效果与性能。自动测试通过不能替代真实设备上的语音和延迟测试。
