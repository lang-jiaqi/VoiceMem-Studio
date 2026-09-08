# Persona

该目录预留“这个 Agent 是谁”的人格 Prompt。当前线上人设仍由现有
`web/harness.py` 提供；这里的文件暂不读取，避免只完成一半的迁移导致实际人设变化。

后续迁移时，人格只定义稳定身份、价值边界和关系方式；回复长短、情绪弧线与接话策略
分别由 `speaking_style/` 和 `turn_taking/` 管理。
