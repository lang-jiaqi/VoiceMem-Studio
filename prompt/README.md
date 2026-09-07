# 本地 Prompt 记录

启动 `web/run.py` 后，每次运行会创建一个时间戳子目录。
每次回复生成使用独立的 `.jsonl` 文件；同一文件按调用顺序记录 LLM 请求和各段 TTS 请求。

- `llm`：实际发送的模型、完整 messages（system、历史、记忆和当前问题）及生成参数。
- `tts` / `breeze_mlx`：每段实际文本、最终 instruct、参考音频路径、完整 ref_text 和生成参数。
- `output_id`：关联该次音频输出；提前生成的请求也保留，不代表最终一定播放。
- 无回复上下文的合成（例如附和预合成）单独存文件，标记 `purpose: unscoped`。

预缓存附和的播放不会重新调用 TTS，因此不会伪造新的 TTS 请求记录。
文件后台写入；正常退出会等待写完，强制杀进程可能丢失末尾尚未写入的记录。
这里只保存 prompt 和允许的生成参数，不保存 API key / Authorization headers。
内容可能包含个人记忆和完整对话，已被 `.gitignore` 排除；不要公开分享。

终端默认精简；加 `--verbose` 恢复完整输出。详细运行日志仍保留在 `results/logs/`。
