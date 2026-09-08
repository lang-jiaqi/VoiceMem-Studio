# Reply modes

该目录预留回复路径选择，当前尚未接入运行时路由：

- `memory_cot.md`：使用记忆，并允许较长推理或工具工作。
- `memory.md`：使用检索记忆，直接组织回复。
- `direct.md`：不使用长期记忆，直接回应当前上下文。

三个文件目前只是接口和 Prompt 占位。后续实现路由时应输出
`memory_cot`、`memory` 或 `direct`，不要在其他模块另造名称。
