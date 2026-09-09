/* 把 VoiceMem 后端的播放状态接到嘴上。
 *
 * 没有 ?ws= 参数就什么都不做——直接双击打开时它还是原来那只独立桌宠。
 *
 * 驱动信号用的是浏览器回传给后端的 playback_checkpoint，不是后端自己的
 * answer_start / answer_done。后者是"生成"的起止，前者是"声音真的在响"的起止：
 * answer_done 发出来的时候，页面那边往往还有几百毫秒到几秒的音频排在缓冲里没播完，
 * 照着它闭嘴，嘴会比声音先停。
 *
 * 播放中 checkpoint 每 200ms 来一条（见 pcm-player-worklet.js 末尾的 _report）。
 * 所以这里不用"开/关"两个状态，而是每来一条就把 talk() 的窗口续到 1 秒后——
 * 声音停了、checkpoint 自然断了，嘴最迟 1 秒内自己合上。丢一两条消息、
 * 后端崩了、网线拔了，都不会留下一张永远在动的嘴。
 */
(() => {
  const url = new URLSearchParams(location.search).get('ws');
  if (!url) return;

  const WINDOW_S = 1;          // 续一秒：比 200ms 的心跳宽裕，又不至于拖出可见的尾巴
  const BACKOFF_MAX = 5000;
  let socket, retries = 0, talking = false;

  // 说话时如果还缩成小点，先展开——不然"有反应"这件事根本看不见。
  // toggle 是开关不是设值，所以得先问一下当前是什么状态。
  const wake = () => { if (document.body.dataset.mode === 'dot') window.pet?.toggle?.(); };

  function speak() {
    wake();
    talking = true;
    window.petRig?.talk(WINDOW_S);
  }

  function hush() {
    if (!talking) return;
    talking = false;
    window.petRig?.stopTalking();
  }

  function handle(message) {
    // 浏览器回传的播放心跳：playing 续窗口，其余状态立刻停。
    if (message.type === 'playback_checkpoint') {
      if (message.state === 'playing') speak();
      else hush();                       // paused（用户插话）/ stalled / drained / interrupted
      return;
    }
    // 接话（听人说话时"嗯"一声）：点个头。
    // 这一声走的是页面里独立的 AudioBufferSource，不进 pcm worklet，所以没有
    // checkpoint 可跟——嘴不会动，只有头点一下，这跟"它不是一轮回复"是一致的。
    if (message.type === 'backchannel') { wake(); window.petRig?.nod(); return; }
    // 后端侧的兜底：被打断时页面不一定还来得及回一条 checkpoint。
    if (message.type === 'answer_interrupt') hush();
  }

  function connect() {
    try { socket = new WebSocket(url); }
    catch { return schedule(); }
    socket.onopen = () => { retries = 0; };
    socket.onmessage = event => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      handle(message);
    };
    // 后端可能比小人先重启（改代码、换端口），所以断了就一直退避重连，
    // 不把这只桌宠变成一次性的。
    socket.onclose = () => { hush(); schedule(); };
    socket.onerror = () => { try { socket.close(); } catch {} };
  }

  function schedule() {
    setTimeout(connect, Math.min(BACKOFF_MAX, 250 * 2 ** retries++));
  }

  connect();
})();
