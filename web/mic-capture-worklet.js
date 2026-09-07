// Browser AEC runs first. This guard only attenuates highly correlated residual
// echo; unrelated near-end speech (including double-talk) passes unchanged.
class VoiceMemMicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameSize = Math.round(sampleRate * .02);
    this.mic = new Float32Array(this.frameSize);
    this.ref = new Float32Array(this.frameSize);
    this.fill = 0;
    this.factor = Math.max(1, Math.round(sampleRate / 8000));
    this.history = [];
    this.suppressed = 0;
    this.frames = 0;
  }
  clean(mic, reference) {
    const k = this.factor, low = [], ref = [];
    for (let i = 0; i + k <= mic.length; i += k) {
      let a = 0, b = 0;
      for (let j = 0; j < k; j++) { a += mic[i+j]; b += reference[i+j]; }
      low.push(a/k); ref.push(b/k);
    }
    this.history.push(...ref);
    const maxLag = Math.round(sampleRate / k * .35);
    if (this.history.length > maxLag + low.length)
      this.history.splice(0, this.history.length - maxLag - low.length);
    let energy = 0;
    for (const x of low) energy += x*x;
    if (energy < low.length * 1e-7) return mic;
    const end = this.history.length - low.length;
    let best = 0, bestLag = 0;
    const score = lag => {
      let dot = 0, power = 0;
      const start = end - lag;
      for (let i = 0; i < low.length; i++) {
        const r = this.history[start+i];
        dot += low[i]*r; power += r*r;
      }
      if (power < low.length * 1e-6) return 0;
      return Math.abs(dot) / Math.sqrt(energy*power);
    };
    for (let lag = 0; lag <= end; lag += 4) {
      const c = score(lag);
      if (c > best) { best = c; bestLag = lag; }
    }
    for (let lag = Math.max(0,bestLag-3); lag <= Math.min(end,bestLag+3); lag++)
      best = Math.max(best, score(lag));
    // Deliberately high: mixed user+speaker speech must not be muted.
    if (best >= .88) {
      for (let i = 0; i < mic.length; i++) mic[i] *= .03;
      this.suppressed++;
    }
    return mic;
  }
  process(inputs, outputs) {
    // The microphone never loops back to the speakers.
    for (const output of outputs) for (const channel of output) channel.fill(0);
    const mic = inputs[0] && inputs[0][0];
    const ref = inputs[1] && inputs[1][0];
    if (!mic) return true;
    for (let i = 0; i < mic.length; i++) {
      this.mic[this.fill] = mic[i];
      this.ref[this.fill++] = ref ? (ref[i] || 0) : 0;
      if (this.fill === this.frameSize) {
        const pcm = this.clean(this.mic, this.ref);
        this.port.postMessage({type:'mic',samples:pcm}, [pcm.buffer]);
        this.mic = new Float32Array(this.frameSize);
        this.fill = 0;
        if (++this.frames % 50 === 0) {
          this.port.postMessage({type:'echo_stats',suppressed:this.suppressed,frames:50});
          this.suppressed = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('voicemem-mic-capture', VoiceMemMicCapture);
