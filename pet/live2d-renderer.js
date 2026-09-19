(function (root, factory) {
  const api = factory(root);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this, root => {
  const CORES = ['assets/live2d/vendor/live2dcubismcore.min.js', 'https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js'];
  const PIXI = 'node_modules/pixi.js/dist/browser/pixi.min.js';
  const DISPLAY = 'node_modules/pixi-live2d-display/dist/cubism4.min.js';

  function script(path, ready) {
    if (ready()) return Promise.resolve();
    const source = new URL(path, location.href).href;
    const existing = document.querySelector(`script[src="${source}"]`);
    if (existing) return new Promise((resolve, reject) => {
      existing.addEventListener('load', resolve, { once: true });
      existing.addEventListener('error', reject, { once: true });
    });
    return new Promise((resolve, reject) => {
      const tag = document.createElement('script');
      tag.src = source; tag.onload = resolve; tag.onerror = () => reject(new Error(`Unable to load ${path}`));
      document.head.appendChild(tag);
    });
  }

  class Live2DRenderer {
    constructor(canvas) {
      this.canvas = canvas; this.app = null; this.model = null; this.path = '';
      this.parameters = Object.create(null); this.armAccents = null; this.active = false; this.error = null;
      this.nativeMotion = false;
      this.expressionTimer = 0; this.expressionEpoch = 0;
      this.currentExpression = ''; this.expressionError = null;
      this.contextLost = false; this.width = 0; this.height = 0; this.resolution = 0;
      this.applyParameters = this.applyParameters.bind(this);
      this.applyArmAccents = this.applyArmAccents.bind(this);
      canvas.addEventListener('webglcontextlost', event => { event.preventDefault(); this.contextLost = true; });
      canvas.addEventListener('webglcontextrestored', () => { this.contextLost = false; if (this.path) this.loadModel(this.path).catch(() => {}); });
    }
    async ensureRuntime() {
      let cause;
      for (const source of CORES) {
        try { await script(source, () => Boolean(root.Live2DCubismCore)); break; }
        catch (error) { cause = error; }
      }
      if (!root.Live2DCubismCore) { const error = new Error(`Cubism Core is missing. Copy live2dcubismcore.min.js to ${CORES[0]}`); error.code = 'CUBISM_CORE_MISSING'; error.cause = cause; throw error; }
      await script(PIXI, () => Boolean(root.PIXI));
      await script(DISPLAY, () => Boolean(root.PIXI?.live2d?.Live2DModel));
    }
    async loadModel(path) {
      if (!path || !String(path).endsWith('.model3.json')) throw new Error('A Cubism .model3.json path is required');
      await this.ensureRuntime();
      this.destroyModel(); this.path = String(path); this.error = null;
      if (!this.app) this.app = new root.PIXI.Application({ view: this.canvas, transparent: true,
        backgroundAlpha: 0, antialias: true, autoStart: false, sharedTicker: false, resolution: 1 });
      try {
        this.model = await root.PIXI.live2d.Live2DModel.from(this.path, { autoUpdate: false, autoInteract: false });
        this.model.autoUpdate = false;
        const internal = this.model.internalModel;
        internal.eyeBlink = null; internal.breath = null;
        internal.motionManager?.on('motionFinish', () => { this.nativeMotion = false; });
        internal.focusController?.focus(0, 0, true);
        // Apply once before physics so hair/cloth can react to head motion, then
        // once after physics so controlled facial parameters remain authoritative.
        internal.on('afterMotionUpdate', this.applyParameters);
        internal.on('beforeModelUpdate', this.applyParameters);
        internal.on('beforeModelUpdate', this.applyArmAccents);
        this.app.stage.addChild(this.model); this.active = true; this.resize();
        return this;
      } catch (error) { this.error = String(error?.message || error); this.destroyModel(); throw error; }
    }
    applyParameters() {
      const core = this.model?.internalModel?.coreModel; if (!core) return;
      const entries = this.nativeMotion
        ? [['ParamMouthOpenY', this.parameters.ParamMouthOpenY]]
        : Object.entries(this.parameters);
      for (const [id, value] of entries) {
        if (!Number.isFinite(value)) continue;
        try { core.setParameterValueById(id, value); } catch {}
      }
    }
    applyArmAccents() {
      const core = this.model?.internalModel?.coreModel;
      if (!core || !this.armAccents || this.nativeMotion) return;
      for (const [id, offset] of Object.entries(this.armAccents)) {
        if (Number.isFinite(offset)) core.addParameterValueById(id, offset);
      }
    }
    setParameters(values) { this.parameters = values || this.parameters; }
    setArmAccents(values) { this.armAccents = values; }
    playMotion(group, index = 0) {
      if (!this.model) return false;
      this.nativeMotion = true;
      this.model.motion(group, index, 3).then(started => { if (!started) this.nativeMotion = false; })
        .catch(() => { this.nativeMotion = false; });
      return true;
    }
    playExpression(name, durationMs = 1800) {
      if (!this.model || typeof name !== 'string' || !name) return false;
      const epoch = ++this.expressionEpoch;
      this.expressionError = null;
      clearTimeout(this.expressionTimer); this.expressionTimer = 0;
      this.model.expression(name).then(started => {
        if (epoch !== this.expressionEpoch || !this.model) return;
        if (!started) { this.expressionError = `Expression unavailable: ${name}`; return; }
        this.currentExpression = name;
        const duration = Math.max(0, Number(durationMs) || 0);
        if (duration) this.expressionTimer = setTimeout(() => {
          if (epoch !== this.expressionEpoch || !this.model) return;
          this.model.internalModel?.expressionManager?.resetExpression();
          this.expressionTimer = 0; this.currentExpression = '';
        }, duration);
      }).catch(error => {
        if (epoch === this.expressionEpoch) this.expressionError = String(error?.message || error);
      });
      return true;
    }
    resize() {
      if (!this.app || !this.model) return;
      const width = Math.max(1, this.canvas.clientWidth), height = Math.max(1, this.canvas.clientHeight);
      const resolution = Math.min(2, root.devicePixelRatio || 1);
      if (width === this.width && height === this.height && resolution === this.resolution) return;
      this.width = width; this.height = height; this.resolution = resolution;
      this.app.renderer.resolution = resolution; this.app.renderer.resize(width, height);
      const bounds = this.model.getLocalBounds();
      const portrait = new URLSearchParams(location.search).get('layout') === 'portrait';
      const scale = portrait
        ? width / Math.max(1, bounds.width) * 1.8
        : Math.min(width / Math.max(1, bounds.width), height / Math.max(1, bounds.height)) * .96;
      this.model.scale.set(scale); this.model.x = width / 2 - (bounds.x + bounds.width / 2) * scale;
      this.model.y = (portrait ? height * 1.84 : height) - (bounds.y + bounds.height) * scale;
    }
    update(deltaMs) {
      if (!this.active || !this.model || this.contextLost) return;
      this.resize(); this.model.update(Math.min(100, deltaMs)); this.app.renderer.render(this.app.stage);
    }
    show() { this.active = true; this.canvas.hidden = false; }
    hide() { this.active = false; this.canvas.hidden = true; }
    hitTest(x, y) {
      if (!this.active || !this.model) return false;
      const rect = this.canvas.getBoundingClientRect();
      return this.model.containsPoint(new root.PIXI.Point(x - rect.left, y - rect.top));
    }
    destroyModel() {
      clearTimeout(this.expressionTimer); this.expressionTimer = 0; this.expressionEpoch++;
      this.currentExpression = ''; this.expressionError = null;
      if (!this.model) return;
      try {
        this.model.internalModel?.off('afterMotionUpdate', this.applyParameters);
        this.model.internalModel?.off('beforeModelUpdate', this.applyParameters);
        this.model.internalModel?.off('beforeModelUpdate', this.applyArmAccents);
        this.app?.stage.removeChild(this.model); this.model.destroy({ children: true, texture: true, baseTexture: true });
      } catch {}
      this.model = null;
      this.armAccents = null;
      this.nativeMotion = false;
    }
    destroy() { this.destroyModel(); this.app?.destroy(false, { children: true }); this.app = null; this.active = false; }
    getStatus() { return { renderer: 'live2d', ready: Boolean(this.model), active: this.active,
      model: this.path, error: this.error, contextLost: this.contextLost,
      expression: this.currentExpression, expressionError: this.expressionError }; }
  }
  return { Live2DRenderer };
});
