(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  const clamp = value => Math.max(0, Math.min(1, value));
  class AudioLipSync {
    constructor(options = {}) {
      this.attack = options.attack ?? .052;
      this.release = options.release ?? .125;
      this.staleMs = options.staleMs ?? 180;
      this.noiseFloor = .003;
      this.rms = 0; this.value = 0; this.target = 0; this.lastTimestamp = 0; this.playing = false;
    }
    setPlaying(active, timestamp = performance.now()) {
      this.playing = Boolean(active); this.lastTimestamp = timestamp;
      if (!active) this.target = 0;
    }
    feed(rms, timestamp = performance.now()) {
      const level = Math.max(0, Math.min(1, Number(rms) || 0));
      this.rms = level; this.lastTimestamp = Number(timestamp) || performance.now(); this.playing = true;
      if (level < this.noiseFloor * 1.8) this.noiseFloor += (level - this.noiseFloor) * .025;
      const gate = Math.max(.004, this.noiseFloor * 2.2);
      const normalized = Math.max(0, level - gate) / Math.max(.02, .105 - gate);
      this.target = normalized > 0 ? clamp(.12 + .88 * Math.sqrt(normalized)) : 0;
    }
    update(dt, now = performance.now()) {
      if (!this.playing || now - this.lastTimestamp > this.staleMs) this.target = 0;
      const tau = this.target > this.value ? this.attack : this.release;
      const desired = this.value + (this.target - this.value) * (1 - Math.exp(-Math.min(.1, dt) / tau));
      const maxDelta = Math.max(.035, dt * 8.5);
      this.value += Math.max(-maxDelta, Math.min(maxDelta, desired - this.value));
      if (this.value < .002 && this.target === 0) this.value = 0;
      return this.value;
    }
    stop(timestamp = performance.now()) { this.setPlaying(false, timestamp); }
    status() { return { rms: this.rms, mouthOpen: this.value, noiseFloor: this.noiseFloor, playing: this.playing }; }
  }
  return { AudioLipSync };
});
