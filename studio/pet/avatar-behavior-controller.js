(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  const STATE = {
    sleeping: { headY: -9, eye: .92, bodyY: -3, focus: .15 },
    idle: { headY: 0, eye: 1, bodyY: 0, focus: .35 },
    listening: { headY: 2, eye: 1, bodyY: 1, focus: .9 },
    thinking: { headY: 1, eye: .92, bodyY: 0, focus: .45 },
    speaking: { headY: 1, eye: 1, bodyY: .5, focus: .8 },
    interrupted: { headY: 2, eye: 1, bodyY: 1, focus: .95 },
    success: { headY: 2, eye: .9, bodyY: .5, focus: .7 },
    error: { headY: -1, eye: .86, bodyY: -1, focus: .65 }
  };
  const EMOTION = {
    neutral: { mouth: 0, brow: .16, cheek: .3 }, happy: { mouth: .65, brow: .24, cheek: .56 },
    curious: { mouth: .1, brow: .34, cheek: .16 }, concerned: { mouth: -.35, brow: .28, cheek: 0 },
    excited: { mouth: .85, brow: .38, cheek: .58 }, sleepy: { mouth: .05, brow: -.12, cheek: .12 }
  };
  const ACTIVE_FRAME_STATES = new Set(['listening', 'speaking', 'interrupted']);
  const clamp01 = x => Math.max(0, Math.min(1, x));
  const ease = x => { x = clamp01(x); return x * x * (3 - 2 * x); };
  const bell = (t, a, b, c) => ease((t - a) / Math.max(.001, b - a)) * (1 - ease((t - b) / Math.max(.001, c - b)));
  const avatarFrameRate = (state, pointerActive = false) =>
    pointerActive || ACTIVE_FRAME_STATES.has(state) ? 60 : 30;

  class AvatarBehaviorController {
    constructor(parameters, options = {}) {
      this.parameters = parameters; this.random = options.random || Math.random;
      this.state = 'sleeping'; this.emotion = 'neutral'; this.emotionIntensity = 0;
      this.time = 0; this.breathPhase = 0; this.breathPeriod = 4.1;
      this.nextBlink = this.range(2.5, 5.8); this.blink = null;
      this.nextGaze = this.range(1.2, 3.2); this.gaze = { x: 0, y: 0, changed: 0 };
      this.pointerGaze = null; this.nextSpeakingGesture = Infinity; this.armAccents = null;
      this.nextBlush = this.range(4, 9); this.blush = null;
      this.gesture = null; this.cooldowns = new Map(); this.speaking = false;
      this.parameters.setLayer('state', {}, { priority: 20, transitionMs: 900 });
    }
    range(a, b) { return a + (b - a) * this.random(); }
    setState(state) {
      if (!STATE[state]) throw new Error(`Unknown avatar state: ${state}`);
      if (state === 'speaking' && this.state !== 'speaking') this.nextSpeakingGesture = this.time + this.range(.12, .3);
      if (state !== 'speaking') this.nextSpeakingGesture = Infinity;
      this.state = state;
      const pose = STATE[state];
      this.parameters.setLayer('state', {
        ParamAngleY: pose.headY, ParamBodyAngleY: pose.bodyY,
        ParamEyeLOpen: pose.eye, ParamEyeROpen: pose.eye
      }, { priority: 20, weight: 1, transitionMs: state === 'sleeping' ? 1200 : 420 });
      this.speaking = state === 'speaking';
    }
    setPointerGaze(x, y) {
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        this.pointerGaze = null;
        this.parameters.clearLayer('pointer-eyes', 320);
        this.parameters.clearLayer('pointer-head', 480);
        return;
      }
      this.pointerGaze = { x: Math.max(-1, Math.min(1, x)), y: Math.max(-1, Math.min(1, y)) };
      this.parameters.setLayer('pointer-eyes', {
        ParamEyeBallX: this.pointerGaze.x * .85, ParamEyeBallY: this.pointerGaze.y * .65,
      }, { priority: 55, transitionMs: 110 });
      this.parameters.setLayer('pointer-head', {
        ParamAngleX: this.pointerGaze.x * 6, ParamAngleY: this.pointerGaze.y * 4,
      }, { priority: 24, transitionMs: 340 });
    }
    setEmotion(emotion, intensity = 1, duration = 0) {
      if (!EMOTION[emotion]) throw new Error(`Unknown avatar emotion: ${emotion}`);
      this.emotion = emotion; this.emotionIntensity = clamp01(intensity);
      const e = EMOTION[emotion];
      this.parameters.setLayer('emotion', {
        ParamMouthForm: e.mouth, ParamBrowLY: e.brow, ParamBrowRY: e.brow,
        ParamCheek: e.cheek
      }, { priority: 35, weight: this.emotionIntensity, transitionMs: 500 });
      this.emotionUntil = duration > 0 ? this.time + duration / 1000 : 0;
    }
    triggerGesture(name, options = {}) {
      const aliases = { backchannel: 'backchannel', nod: 'nod', tilt: 'tilt-smile',
        'tilt-smile': 'tilt-smile', sway: 'sway', surprise: 'surprise' };
      const kind = aliases[String(name).toLowerCase()];
      if (!kind || this.state === 'sleeping') return false;
      const now = this.time, cooldown = Number(options.cooldown ?? (kind === 'backchannel' ? 1.8 : 4));
      if (this.gesture || now < (this.cooldowns.get(kind) || 0)) return false;
      const durations = { nod: 1.35, backchannel: 1.15, 'tilt-smile': 2.8, sway: 2.2, surprise: .8 };
      this.gesture = { name: kind, start: now, duration: durations[kind], amplitude: Number(options.amplitude ?? 1) };
      this.cooldowns.set(kind, now + cooldown); return true;
    }
    update(dt) {
      dt = Math.max(0, Math.min(.1, Number(dt) || 0)); this.time += dt;
      if (this.emotionUntil && this.time >= this.emotionUntil) { this.emotionUntil = 0; this.setEmotion('neutral', 1); }
      this.breathPhase += dt * Math.PI * 2 / this.breathPeriod;
      if (this.breathPhase >= Math.PI * 2) { this.breathPhase %= Math.PI * 2; this.breathPeriod = this.range(3.2, 5); }
      const idleScale = this.state === 'sleeping' ? .75 : this.state === 'listening' ? .68 : 1.15;
      this.parameters.setLayer('idle', {
        ParamBreath: .5 + .5 * Math.sin(this.breathPhase),
        ParamBodyAngleZ: Math.sin(this.time * .43) * 1.1 * idleScale,
        ParamAngleZ: Math.sin(this.time * .31 + .8) * .85 * idleScale
      }, { priority: 5, transitionMs: 550 });

      if (this.time >= this.nextBlink && !this.blink) {
        this.blink = { start: this.time, double: this.random() < .12 };
      }
      if (this.blink) {
        const age = this.time - this.blink.start, second = this.blink.double ? Math.max(0, 1 - Math.abs(age - .31) / .095) : 0;
        const close = Math.max(Math.max(0, 1 - Math.abs(age - .095) / .095), second);
        this.parameters.setLayer('blink', { ParamEyeLOpen: 1 - close, ParamEyeROpen: 1 - close },
          { priority: 80, transitionMs: 25 });
        if (age > (this.blink.double ? .44 : .22)) {
          this.blink = null; this.parameters.clearLayer('blink', 45); this.nextBlink = this.time + this.range(2.5, 6.5);
        }
      }

      if (this.time >= this.nextGaze && !this.gesture) {
        const focus = STATE[this.state].focus;
        this.gaze = { x: this.range(-.65, .65) * (1 - focus), y: this.range(-.4, .35) * (1 - focus), changed: this.time };
        this.nextGaze = this.time + this.range(1.4, 4);
      }
      const gazeAge = this.time - this.gaze.changed;
      this.parameters.setLayer('gaze-eye', { ParamEyeBallX: this.gaze.x, ParamEyeBallY: this.gaze.y }, { priority: 25, transitionMs: 110 });
      if (gazeAge > .12) this.parameters.setLayer('gaze-head', {
        ParamAngleX: this.gaze.x * 14,
        ParamAngleY: STATE[this.state].headY + this.gaze.y * 9
      }, { priority: 22, transitionMs: 420 });
      if (gazeAge > .28) this.parameters.setLayer('gaze-body', {
        ParamBodyAngleX: this.gaze.x * 3.6 + Math.sin(this.time * .37) * .5 * idleScale
      }, { priority: 8, transitionMs: 520 });

      if (this.time >= this.nextBlush && !this.blush) this.blush = { start: this.time, duration: this.range(3.2, 5.2) };
      if (this.blush) {
        const age = this.time - this.blush.start;
        const warmth = bell(age, 0, this.blush.duration * .38, this.blush.duration);
        this.parameters.setLayer('blush', {
          ParamCheek: clamp01(EMOTION[this.emotion].cheek + .32 * warmth)
        }, { priority: 40, transitionMs: 500 });
        if (age >= this.blush.duration) {
          this.blush = null; this.parameters.clearLayer('blush', 650);
          this.nextBlush = this.time + this.range(10, 22);
        }
      }

      if (this.speaking) this.parameters.setLayer('speaking-rhythm', {
        ParamAngleX: Math.sin(this.time * 2.15) * .7, ParamBodyAngleY: Math.sin(this.time * 1.35) * 1.3
      }, { priority: 21, transitionMs: 240 });
      else this.parameters.clearLayer('speaking-rhythm', 280);

      if (this.speaking && this.time >= this.nextSpeakingGesture) {
        const choices = ['nod', 'sway', 'tilt-smile'];
        const started = this.triggerGesture(choices[Math.min(choices.length - 1, Math.floor(this.random() * choices.length))],
          { cooldown: 0, amplitude: this.range(.85, 1) });
        this.nextSpeakingGesture = this.time + (started ? this.range(3.6, 6.4) : .4);
      }

      if (this.gesture) this.updateGesture();
    }
    updateGesture() {
      const g = this.gesture, t = this.time - g.start, a = g.amplitude;
      let values = {};
      if (g.name === 'nod' || g.name === 'backchannel') {
        const down = bell(t, .12, .55, 1.18), anticipation = bell(t, 0, .11, .25);
        values = { ParamAngleY: a * (anticipation * 3 - down * (g.name === 'backchannel' ? 9 : 15)),
          ParamBodyAngleY: -down * 5.5 * a, ParamBodyAngleX: down * 2.5 * a,
          ParamEyeBallY: -down * .12 };
        this.armAccents = { Param47: down * .27 * a, Param50: down * .27 * a };
      } else if (g.name === 'tilt-smile') {
        const w = bell(t, .14, .72, 2.75);
        values = { ParamAngleZ: -12 * a * w, ParamBodyAngleZ: -7 * a * w,
          ParamBodyAngleX: -3.5 * a * w,
          ParamEyeLOpen: 1 - w, ParamEyeROpen: 1 - w,
          ParamEyeLSmile: w, ParamEyeRSmile: w, ParamMouthForm: .9 * w, ParamCheek: .55 * w };
        this.armAccents = { Param47: .32 * a * w, Param48: .22 * a * w,
          Param50: -.22 * a * w, Param51: -.18 * a * w };
      } else if (g.name === 'sway') {
        const w = bell(t, .1, .8, 2.15);
        values = { ParamBodyAngleX: 8.5 * a * w, ParamBodyAngleZ: -7 * a * w,
          ParamAngleZ: 4 * a * w };
        this.armAccents = { Param47: .4 * a * w, Param48: .28 * a * w,
          Param50: -.36 * a * w, Param51: -.24 * a * w };
      } else {
        const w = bell(t, 0, .12, .78);
        values = { ParamAngleY: 5 * w, ParamEyeLOpen: 1, ParamEyeROpen: 1,
          ParamBrowLY: .65 * w, ParamBrowRY: .65 * w, ParamMouthOpenY: .25 * w };
        this.armAccents = null;
      }
      this.parameters.setLayer('gesture', values, { priority: 65, transitionMs: 45 });
      if (t >= g.duration) {
        this.gesture = null; this.armAccents = null; this.parameters.clearLayer('gesture', 240);
      }
    }
    status() { return { state: this.state, emotion: this.emotion, gesture: this.gesture?.name || null }; }
  }
  return { AvatarBehaviorController, AVATAR_STATES: Object.keys(STATE), AVATAR_EMOTIONS: Object.keys(EMOTION),
    avatarFrameRate };
});
