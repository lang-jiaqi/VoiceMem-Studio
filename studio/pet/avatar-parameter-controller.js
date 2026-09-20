(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, () => {
  const DEFAULTS = {
    ParamAngleX: [-30, 30, 0, .22], ParamAngleY: [-30, 30, 0, .22], ParamAngleZ: [-30, 30, 0, .24],
    ParamBodyAngleX: [-10, 10, 0, .38], ParamBodyAngleY: [-10, 10, 0, .4], ParamBodyAngleZ: [-10, 10, 0, .4],
    ParamEyeLOpen: [0, 1, 1, .055], ParamEyeROpen: [0, 1, 1, .055],
    ParamEyeLSmile: [0, 1, .08, .09], ParamEyeRSmile: [0, 1, .08, .09],
    ParamEyeBallX: [-1, 1, 0, .1], ParamEyeBallY: [-1, 1, 0, .1],
    ParamBrowLY: [-1, 1, 0, .22], ParamBrowRY: [-1, 1, 0, .22],
    ParamBrowLAngle: [-1, 1, 0, .22], ParamBrowRAngle: [-1, 1, 0, .22],
    ParamMouthOpenY: [0, 2.1, 0, .065], ParamMouthForm: [-1, 1, 0, .18],
    ParamCheek: [0, 1, .3, .24], ParamBreath: [0, 1, .5, .3]
  };
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function smoothDamp(state, target, smoothTime, maxSpeed, dt) {
    smoothTime = Math.max(.0001, smoothTime);
    const omega = 2 / smoothTime, x = omega * dt;
    const decay = 1 / (1 + x + .48 * x * x + .235 * x * x * x);
    const original = target;
    let change = state.current - target;
    const maxChange = maxSpeed * smoothTime;
    change = clamp(change, -maxChange, maxChange);
    target = state.current - change;
    const temp = (state.velocity + omega * change) * dt;
    state.velocity = (state.velocity - omega * temp) * decay;
    let output = target + (change + temp) * decay;
    if ((original - state.current > 0) === (output > original)) {
      output = original;
      state.velocity = 0;
    }
    state.current = output;
  }

  class AvatarParameterController {
    constructor(definitions = DEFAULTS) {
      this.parameters = new Map();
      this.layers = new Map();
      this.output = Object.create(null);
      for (const [id, [min, max, initial, smoothing]] of Object.entries(definitions)) {
        this.parameters.set(id, { id, current: initial, target: initial, velocity: 0,
          weight: 1, priority: 0, min, max, smoothing, maxSpeed: (max - min) * 8 });
        this.output[id] = initial;
      }
    }
    setLayer(name, values, options = {}) {
      let layer = this.layers.get(name);
      if (!layer) {
        layer = { name, values: Object.create(null), weight: 0, targetWeight: 1,
          priority: 0, transition: .35, remove: false };
        this.layers.set(name, layer);
      }
      Object.assign(layer.values, values || {});
      layer.priority = Number(options.priority ?? layer.priority);
      layer.targetWeight = clamp(Number(options.weight ?? 1), 0, 1);
      layer.transition = Math.max(.016, Number(options.transitionMs ?? layer.transition * 1000) / 1000);
      layer.remove = false;
      return layer;
    }
    clearLayer(name, transitionMs = 300) {
      const layer = this.layers.get(name);
      if (layer) { layer.targetWeight = 0; layer.transition = Math.max(.016, transitionMs / 1000); layer.remove = true; }
    }
    snap(values = {}) {
      for (const [id, value] of Object.entries(values)) {
        const state = this.parameters.get(id); if (!state) continue;
        state.current = state.target = clamp(Number(value), state.min, state.max); state.velocity = 0;
      }
    }
    update(deltaSeconds) {
      const dt = clamp(Number(deltaSeconds) || 0, 0, .1);
      const layers = [...this.layers.values()].sort((a, b) => a.priority - b.priority);
      for (const layer of layers) {
        const response = 1 - Math.exp(-dt * 4.6 / layer.transition);
        layer.weight += (layer.targetWeight - layer.weight) * response;
        if (layer.remove && layer.weight < .001) this.layers.delete(layer.name);
      }
      for (const state of this.parameters.values()) {
        let target = DEFAULTS[state.id]?.[2] ?? 0, weight = 0, priority = -Infinity;
        for (const layer of layers) {
          if (!(state.id in layer.values) || layer.weight <= .0001) continue;
          const value = clamp(Number(layer.values[state.id]), state.min, state.max);
          if (layer.priority > priority) { target = value; weight = layer.weight; priority = layer.priority; }
          else if (layer.priority === priority) {
            const total = weight + layer.weight;
            target = total ? (target * weight + value * layer.weight) / total : value; weight = clamp(total, 0, 1);
          }
        }
        const base = DEFAULTS[state.id]?.[2] ?? 0;
        state.target = clamp(base + (target - base) * weight, state.min, state.max);
        state.weight = weight; state.priority = priority;
        smoothDamp(state, state.target, state.smoothing, state.maxSpeed, dt);
        state.current = clamp(state.current, state.min, state.max);
        this.output[state.id] = state.current;
      }
      return this.output;
    }
    status() {
      const result = {};
      for (const state of this.parameters.values()) result[state.id] = { ...state };
      return result;
    }
  }
  return { AvatarParameterController, smoothDamp, AVATAR_PARAMETER_DEFAULTS: DEFAULTS };
});
