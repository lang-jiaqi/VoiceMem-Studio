class VoiceMemPCMPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.bufferedFrames = 0;
    this.started = false;
    this.draining = false;
    this.paused = false;
    this.outputId = "";
    this.sourceSampleRate = 24000;
    this.renderedSamples = 0;
    this.baseTargetFrames = Math.round(sampleRate * 0.08);
    this.targetFrames = this.baseTargetFrames;
    this.maxTargetFrames = Math.round(sampleRate * 0.32);
    this.stableFrames = 0;
    this.reportFrames = 0;

    // ── 底噪（comfort noise）────────────────────────────────────────────────
    // 谁都不出声的时候输出纯数字静音，听感是"电话挂断了"——人对绝对静音的解读
    // 不是"在等你"，是"断了"。电话系统几十年前就靠 comfort noise 解决这个：
    // 填一层极低的房间底噪，线路就一直"活着"。
    //
    // 用粉噪不用白噪：白噪是嘶嘶的电子声，听得出是人造的；粉噪能量按 1/f 分布，
    // 接近真实房间的本底，耳朵会自动忽略它。
    //
    // 助手说话时不掐断、只压低：开口瞬间掐掉底噪反而制造一个可闻的"门"开合声，
    // 那正是底噪本来要消除的东西。
    this.comfortGain = 0;          // 当前实际增益（一阶平滑，避免咔哒声）
    this.comfortTarget = 0;        // 目标增益：空闲时满、出声时压低
    this.comfortLevel = 0;         // 配置的峰值幅度（0 = 关掉）
    this.comfortDuck = 0.35;       // 出声时压到多少
    this.pink = [0, 0, 0];         // 粉噪滤波器状态（Kellett 三极点近似）
    //: 回复音量。1 = 原样。TTS 出来的电平本来就偏保守，配上笔记本外放常常偏小。
    //: 超过 1 要软限幅，见 process() 末尾——直接乘会削波，听感是"破音"。
    this.gain = 1;

    this.port.onmessage = (event) => {
      const message = event.data || {};
      if (message.type === "config") {
        const seconds = Math.max(0.04, Math.min(0.32, Number(message.prebuffer) || 0.08));
        this.baseTargetFrames = Math.round(sampleRate * seconds);
        this.targetFrames = Math.max(this.baseTargetFrames, this.targetFrames);
        if (message.comfortDb !== undefined) this._setComfort(message.comfortDb);
        return;
      }
      if (message.type === "comfort") {          // 运行时调，用来现场试音量
        this._setComfort(message.db);
        return;
      }
      if (message.type === "gain") {             // 回复音量
        this.gain = Math.max(0, Math.min(4, Number(message.value) || 1));
        return;
      }
      if (message.type === "start") {
        this._clear(false);
        this.outputId = String(message.outputId || "");
        this.sourceSampleRate = Math.max(1, Number(message.sampleRate) || 24000);
        return;
      }
      if (message.type === "audio" && message.samples) {
        const samples = message.samples;
        if (samples.length) {
          this.queue.push({
            samples,
            sourceFrames: Math.max(0, Number(message.sourceFrames) || samples.length),
          });
          this.bufferedFrames += samples.length;
        }
        return;
      }
      if (message.type === "drain") {
        this.draining = true;
        if (!this.bufferedFrames) this._drained();
        return;
      }
      if (message.type === "pause") {
        this.paused = true;
        this._report("paused");
        return;
      }
      if (message.type === "resume") {
        this.paused = false;
        this._report("resumed");
        return;
      }
      if (message.type === "clear") {
        this._clear(message.reason === "interrupted", message.reason || "reset");
      }
    };
  }

  _setComfort(db) {
    // 用 dBFS 配而不是线性幅度：这个量要贴着听感调，而听感是对数的。
    // -48 dBFS 大约是"安静房间里刚好能察觉"，关掉传 null / -Infinity。
    const v = Number(db);
    this.comfortLevel = (db === null || db === undefined || !isFinite(v))
      ? 0 : Math.min(0.05, Math.pow(10, v / 20));
  }

  _comfort(output, written) {
    // 每一条 process() 的返回路径都要经过这里——底噪的意义就在于"任何时候都在"，
    // 漏掉一条路径（暂停、缓冲中、欠载）就会在那个状态下突然断掉，比没有更糟。
    if (this.comfortLevel <= 0) return;
    this.comfortTarget = written > 0
      ? this.comfortLevel * this.comfortDuck    // 正在出声：压低，但不掐断
      : this.comfortLevel;
    // 一阶平滑，约 50ms 时间常数：增益跳变会有咔哒声。
    const a = Math.exp(-1 / (sampleRate * 0.05));
    for (let i = 0; i < output.length; i++) {
      this.comfortGain = this.comfortTarget + (this.comfortGain - this.comfortTarget) * a;
      const white = Math.random() * 2 - 1;
      // Kellett 的经济版粉噪：三个一阶低通并联，比 Voss-McCartney 便宜，
      // 在 -48dBFS 这种量级上听不出区别。
      this.pink[0] = 0.99765 * this.pink[0] + white * 0.0990460;
      this.pink[1] = 0.96300 * this.pink[1] + white * 0.2965164;
      this.pink[2] = 0.57000 * this.pink[2] + white * 1.0526913;
      const pink = (this.pink[0] + this.pink[1] + this.pink[2] + white * 0.1848) / 4;
      output[i] += pink * this.comfortGain;
    }
  }

  _clear(notify, state = "reset") {
    if (notify) this._report(state);
    this.queue.length = 0;
    this.offset = 0;
    this.bufferedFrames = 0;
    this.started = false;
    this.draining = false;
    this.paused = false;
    this.stableFrames = 0;
    this.outputId = "";
    this.renderedSamples = 0;
  }

  _drained() {
    this._report("drained");
    this._clear(false);
    this.targetFrames = this.baseTargetFrames;
  }

  _report(type) {
    this.port.postMessage({
      type,
      bufferedMs: Math.round((this.bufferedFrames / sampleRate) * 1000),
      targetMs: Math.round((this.targetFrames / sampleRate) * 1000),
      outputId: this.outputId,
      renderedSamples: Math.round(this.renderedSamples),
      sampleRate: this.sourceSampleRate,
    });
  }

  process(_inputs, outputs) {
    const output = outputs[0] && outputs[0][0];
    if (!output) return true;
    output.fill(0);

    // 候选插话期间输出静音并保留队列，resume 后从暂停位置继续。
    // 底噪照旧：这一刻正是用户刚开口、助手刚闭嘴，最不该让人觉得"断了"。
    if (this.paused) { this._comfort(output, 0); return true; }

    if (!this.started) {
      if (!this.bufferedFrames) { this._comfort(output, 0); return true; }
      if (!this.draining && this.bufferedFrames < this.targetFrames) {
        this._comfort(output, 0); return true;
      }
      this.started = true;
      this._report("started");
    }

    let written = 0;
    while (written < output.length && this.queue.length) {
      const head = this.queue[0];
      const count = Math.min(
        output.length - written, head.samples.length - this.offset);
      output.set(head.samples.subarray(this.offset, this.offset + count), written);
      written += count;
      this.offset += count;
      this.bufferedFrames -= count;
      this.renderedSamples += count * head.sourceFrames / head.samples.length;
      if (this.offset >= head.samples.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }

    if (written < output.length && !this.bufferedFrames) {
      if (this.draining) {
        this._drained();
      } else {
        // Increase the jitter buffer after an underrun.
        this.started = false;
        this.stableFrames = 0;
        this.targetFrames = Math.min(
          this.maxTargetFrames,
          this.targetFrames + Math.round(sampleRate * 0.02),
        );
        this._report("underflow");
      }
    } else {
      this.stableFrames += written;
      // Reduce the buffer after stable playback.
      if (this.stableFrames >= sampleRate * 5 && this.targetFrames > this.baseTargetFrames) {
        this.targetFrames = Math.max(
          this.baseTargetFrames,
          this.targetFrames - Math.round(sampleRate * 0.01),
        );
        this.stableFrames = 0;
      }
    }

    if (this.gain !== 1) {
      // 软限幅（tanh 近似）：直接乘会削波。x/(1+|x|/k) 在小信号上几乎是线性的，
      // 大信号平滑压到 ±1，不会出现硬切那种破音。
      for (let i = 0; i < written; i++) {
        const v = output[i] * this.gain;
        output[i] = v / (1 + Math.abs(v) * 0.3);
      }
    }

    this._comfort(output, written);

    this.reportFrames += output.length;
    if (this.reportFrames >= sampleRate / 5) {
      this.reportFrames = 0;
      this._report("buffer");
    }
    return true;
  }
}

registerProcessor("voicemem-pcm-player", VoiceMemPCMPlayer);
