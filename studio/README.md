# Studio

```text
studio/
  core/
    core.py                  # 启动服务、监听、等待、叫停、路由、回复
    voiceagent.py            # 组合组件，持有应用状态
    voicemem.py              # 唯一 VoiceMem 初始化与原生流式入口
    utils/
      asr/                   # component.py + initialize.py
      eot/
      llm/                   # 默认 DeepSeek；保留本地回复与缓存
      tts/                   # Breeze、初始化、快速缓存、语气协议
      conversation/          # 会话逻辑与状态初始化
      reply_modes/
      turn_taking/
      models/                # 权重完整性检查与下载清单
      startup/               # 环境检查与严格预热
      ...                    # 捕获、打断、回放、入库、展示等独立组件
  models/                    # 只放权重，不放 Python 实现
  harness/
    persona/policy.py
    speaking_style/policy.py
    reply_modes/policy.py
    turn_taking/policy.py
  web/                       # 浏览器、HTTP/WebSocket、AudioWorklets
  apps/
```

在仓库根目录使用原生 Apple Silicon 的 Python 3.12 环境。首次安装 Studio 依赖：

```bash
python -m pip install -e '.[studio]'
```

启动：

```bash
python web/run.py --mode llm_tts --space demo-zh --lang zh --confirm_ms 200 --verbose
```

打开 `http://localhost:8787`。`python -m studio` 使用相同入口。
可用 `--llm local`、`--llm openai` 或 `--mode realtime` 选择已有模式。

启动依次检查凭据、依赖版本、资源、四份 policy 和模型清单；缺项集中打印并退出。
只从启动进程的环境读取 `DEEPSEEK_API_KEY` 和 `OPENAI_API_KEY`，不打印值。
DeepSeek 用于默认回复，OpenAI 仍用于 VoiceMem 记忆抽取及后台整理。
其它 Studio 参数由组件初始化文件固定，不再要求配置环境变量。
加 `--check` 只检查，不下载、不打开 Memory Space。

检查通过后，完整的原有权重以本地链接复用到 `studio/models/`，缺少的权重自动下载；
中断后再次启动会复用下载缓存。选中的模型预热失败会阻止服务启动。
参考录音和审核后的附和素材保留在 `voice/`，不会自动生成替代声音。

当前入口在同一进程中先初始化 VoiceMem，再开放 Studio 服务；没有独立的记忆 RPC
服务。`core/voicemem.py` 直接调用原生流式接口，保留线程、取消与投机检索边界。
Memory Spaces、录音、日志和原有模型目录保留。旧 provider 导入仅兼容转发。

Studio prompt 只保留一份，不再按语言复制；`--lang` 继续控制 ASR 和 Memory Space。
VoiceMem 库自身的多语言默认 prompt 保留。四秒追问只针对明确未完成的残句；
“不是”“不对”“我有一个问题”等不触发。续说会合并，断开和切换空间会取消或丢弃。

离线回归覆盖状态和协议；真实音色、麦克风与端到端延时需要完整原生模型环境验收。

Studio uses `transformers==5.16.1` and `huggingface-hub>=1.5,<2` with
`mlx-audio==0.5.1`. Install the complete `studio` extra together; do not use
`--no-deps` to bypass dependency constraints.

情绪识别使用共享的 SenseVoiceSmall CPU 实例，保留情绪标签，不再生成多模态情绪原因。
