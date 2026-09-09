# Reply modes

该目录包含一条三级回复路径：

- `direct.md`（instant）：不使用长期记忆，不开启长推理，直接回应当前上下文。
- `memory.md`（mem）：检索并注入长期记忆，直接组织回复，不开启 CoT。
- `memory_cot.md`（mem+cot）：检索并注入长期记忆，同时开启较长推理。

三个文件仍是回复路径的 Prompt 占位；路径标识固定为
`memory_cot`、`memory` 或 `direct`。

`thinking.py` 在最终 ASR 后输出内部标签 `fast`、`medium` 或 `slow`，分别对应
上面的三条路径。它先应用少量高置信产品规则：明确求解／证明走 mem+cot，寒暄／
助手自我介绍／故事走 instant，Turn Gate 已确认依赖个人历史的走 mem，极短残片走
instant；剩余模糊句才交给本地 Qwen3-0.6B。中文输入使用全中文分类 Prompt。

最终分类是权威结果：instant 清除投机检索结果；mem 和 mem+cot 若没有可复用的
投机结果则补做检索。只有 mem+cot 会向下游 provider 请求长推理，并允许
turn-taking 使用长工作 filler。
