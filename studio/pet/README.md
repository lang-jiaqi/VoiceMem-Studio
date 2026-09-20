# VoiceMem 桌面伙伴

置顶的 Electron 桌宠，使用标准 Cubism model3/moc3 Live2D 模型。

桌宠面向 Windows/macOS 本地桌面；Linux/WSL 后端只广播事件，不自动创建桌宠窗口。
使用 [Studio 桌面 App](../apps/README.md) 时，桌宠已内置，无需在本目录安装依赖或单独启动。
以下命令仍适用于独立桌宠；App 打包直接复用这里的渲染与动画文件。

## 启动

```bash
cd studio/pet
npm install
npm start
```

Mac 原生后端在启用自动桌宠时会启动它并连接 `/ws-pet`；Linux/WSL 只广播事件。
通过 App 使用时由 App 管理桌宠，Mac 后端设置 `STUDIO_DESKTOP_PET=0` 可避免重复窗口。

默认加载 `assets/live2d/rattan/rattan.model3.json`（白藤），以 3:4 竖屏半身构图显示；
通话窗口使用 `assets/scene/call-background.png` 作为背景。指定其他正式 Live2D 模型：

```bash
npm start -- --model=assets/live2d/rattan/rattan.model3.json
```

先从 Live2D 官网 SDK 中取得 `live2dcubismcore.min.js`，放到
`assets/live2d/vendor/`。模型包格式及许可注意事项见
[`assets/live2d/README.md`](assets/live2d/README.md)。

正常与 VoiceMem 一起运行时不需要单独启动：打开 `http://127.0.0.1:8787/?pet=1`，后端会启动桌宠并连接 `/ws-pet`。

正式模型可通过环境变量交给 Studio 自动启动：

```bash
export STUDIO_PET_MODEL=assets/live2d/rattan/rattan.model3.json
python -m studio --llm deepseek --backchannel --port 8787
```

## 行为

- App 内桌宠的“开始对话”按钮控制主 App 当前页面的语音会话，不创建第二条对话；首次麦克风授权由 App 完成。独立桌宠无法控制 App，会提示先使用 Studio App。
- 按住人物或背景可拖动窗口；拖动任意一个角可等比例缩放，范围为 40%–150%。可通过托盘菜单恢复原始大小。
- 位置和大小自动保存。窗口保持置顶，打开 App 主界面时仍由你手动摆放，可缩小或移开以免遮住字幕。
- 双击人物或背景、或按 Escape 收成小点。收起后不会被语音事件自动展开；点击小点再次展开并恢复之前的大小。
- 助手真实输出到播放总线的音频 RMS 驱动嘴型；停止、暂停或断流时平滑闭嘴。
- 助手说话期间按间隔随机出现点头、轻摆和歪头动作，避免每帧重复触发；鼠标位于窗口内时眼睛会平滑跟随，离开后恢复自然视线。
- backchannel 使用轻微点头，继续由分层参数控制器平滑驱动。
- 明确识别为“悲伤 / sad”时，才会短暂使用流泪或冒汗表情。
- 启动时显示休息状态；开始对话或检测到人声时切换到交互状态，20 秒无人声后恢复休息状态。

## 文件

| 文件 | 作用 |
| --- | --- |
| `main.cjs` | Electron 窗口、托盘和桌宠进程生命周期 |
| `avatar-controller.js` | 稳定公开 API、动画帧循环与 Live2D 动作映射 |
| `live2d-renderer.js` | Pixi/Cubism 模型加载、参数写入、缩放和 WebGL 恢复 |
| `avatar-parameter-controller.js` | 分层优先级、边界限制与帧率无关平滑 |
| `avatar-behavior-controller.js` | 状态、情绪、眨眼、视线、呼吸和手势节奏 |
| `audio-lip-sync.js` | 真实播放音量的门限、攻击和释放 |
| `assets/scene/call-background.png` | 通话窗口背景 |
| `voicemem-link.js` | 将 `/ws-pet` 播放及会话事件映射为人物动作 |
| `renderer.js` | 点击、拖动、收起等界面交互 |

`window.avatar` 提供 `loadModel()`、`setState()`、`setEmotion()`、`setSpeaking()`、
`feedAudioLevel()`、`triggerGesture()`、`sleep()`、`wake()`、`getStatus()` 和 `destroy()`。
