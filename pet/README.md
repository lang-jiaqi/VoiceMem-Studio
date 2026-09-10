# 桌面小人（雾铃 Noctelle）

飘在所有窗口最上层的 Live2D 桌宠，助手出声时跟着动嘴，不出声时安静待机。

来源是 `Noctelle-Live2D-0.3.1-Windows` 那个发行包里的 `resources/app.asar`——
Electron 的 asar 只是打包容器不是编译，解开就是完整源码。这份是解出来之后去掉
Windows 运行时、只保留跨平台部分的版本，另外加了 `voicemem-link.js`。

渲染栈是 PixiJS 6.5.10 + pixi-live2d-display 0.4.0 + 官方 Cubism Core 5.1.0，
纯网页技术，跟操作系统无关；那个 `.exe` 只是打包目标选了 Windows 而已。

## 跑起来

第一次要装 Electron（约 200MB，**在 macOS 的终端里跑，不要在别的环境装**，
不然装进来的是别的平台的二进制）：

```bash
cd pet && npm install
```

单独试一下（不连 VoiceMem，就是原来那只桌宠）：

```bash
npm start
```

跟着 VoiceMem 一起用——**正常不用手动跑这条**，后端会自己拉起来：

```bash
npm run start:linked
```

## 跟 VoiceMem 怎么接上的

带 `?pet=1` 打开 demo 页就行：

```
http://127.0.0.1:8787/?pet=1
```

- 后端在 `/` 这个路由上看到 `pet=1` → 拉起小人（已经在跑就跳过，永远只有一只）
- 小人连上 `/ws-pet`，这是个只读广播口，不会开新会话
- 后端把播放状态抄给它：助手出声 → `petRig.talk()`，停 → `petRig.stopTalking()`
- **生死跟后端进程绑定**：关标签页、刷新、开十个标签页都不动它，后端退出才跟着关

不带 `?pet=1` 打开就是原来的页面，一点没变。

嘴型跟的是 `playback_checkpoint`——页面音频 worklet 回传的真实播放状态，
不是后端的 `answer_start`/`answer_done`。后者是"生成"的起止，`answer_done`
发出去的时候页面往往还有一截音频没播完，照着它闭嘴会比声音先停。

想换端口/换成打包好的 .app：

```bash
VOICEMEM_PET_CMD="open -a /Applications/Noctelle.app --args" python -m studio
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `main.cjs` | Electron 主进程：无边框透明置顶窗口、托盘、拖拽 |
| `rig.js` | Live2D 控制核心，对外暴露 `window.petRig` |
| `voicemem-link.js` | **新增**：连后端 `/ws-pet`，把播放状态变成嘴型 |
| `renderer.js` | 交互（点击唤醒、拖动、双击收起、按钮） |
| `tilted-smile.js` | "歪头笑"那 4 秒动作的缓动曲线 |
| `models/sit`, `models/lie` | 两套 Cubism 模型，各带 blink/idle/nod/shake |

`window.petRig` 的接口：`show(pose)` / `hide()` / `talk(seconds)` /
`stopTalking()` / `tilt()` / `status()`。

## 还没做的

- **表情/姿态跟情绪联动**：后端 `memory_hits` 里已经有 `emotion` 字段，
  但现在只有 sit/lie 两个姿势，要更多状态得用 PSD2Live 再生成几套模型挂进
  `models/`，`show()` 本身是通用的，加模型不用改代码
- `models/` 里 `nod` / `shake` 两个 motion 文件生成了但没接线——`rig.js` 现在
  是每帧手写参数、没有走 Cubism 的 MotionManager
- 原始 PSD 和 Cubism 工程在 `../live2d-work`，不在这个目录里；改美术要去那边
