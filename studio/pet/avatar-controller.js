(() => {
  const canvas = document.querySelector('#live');
  const parameters = new AvatarParameterController();
  const behavior = new AvatarBehaviorController(parameters);
  const lipSync = new AudioLipSync();
  const renderer = new Live2DRenderer(canvas);
  let pose = 'lie', active = false, modelError = null;
  let frame = 0, last = performance.now(), frames = 0, fpsAt = last, fps = 0, modelPath = '', action = '';

  async function loadModel(path) {
    modelPath = String(path || ''); modelError = null;
    if (!modelPath) return false;
    try {
      await renderer.loadModel(modelPath);
      if (active) renderer.show(pose); return true;
    } catch (error) {
      modelError = { code: error.code || 'MODEL_LOAD_FAILED', message: String(error.message || error) };
      console.error('[avatar] Live2D unavailable:', modelError.message); return false;
    }
  }
  function tick(now) {
    if (!active) return;
    const elapsed = now - last;
    const interval = 1000 / avatarFrameRate(behavior.state, Boolean(behavior.pointerGaze));
    if (elapsed + .5 < interval) { frame = requestAnimationFrame(tick); return; }
    const dt = Math.min(.1, Math.max(0, elapsed / 1000)); last = now;
    behavior.update(dt);
    const mouthOpen = lipSync.playing ? lipSync.update(dt, now) : 0;
    parameters.setLayer('lip-sync', { ParamMouthOpenY: mouthOpen }, { priority: 100, transitionMs: 16 });
    const values = parameters.update(dt); renderer.setParameters(values);
    renderer.setArmAccents(behavior.armAccents); renderer.update(dt * 1000);
    frames++; if (now - fpsAt >= 1000) { fps = frames * 1000 / (now - fpsAt); frames = 0; fpsAt = now; }
    frame = requestAnimationFrame(tick);
  }
  async function show(nextPose = pose) {
    pose = nextPose; active = true;
    if (pose === 'lie') behavior.setState('sleeping');
    else if (behavior.state === 'sleeping') behavior.setState('idle');
    await renderer.show(pose); last = performance.now(); cancelAnimationFrame(frame); frame = requestAnimationFrame(tick);
    window.dispatchEvent(new Event('pet-ready'));
  }
  function hide() { active = false; cancelAnimationFrame(frame); lipSync.stop(); behavior.setPointerGaze(null, null); renderer.hide(); }
  const avatar = {
    loadModel,
    setState(state) { behavior.setState(state); },
    setEmotion(name, intensity = 1, duration = 0) { behavior.setEmotion(name, intensity, duration); },
    setSpeaking(value) { lipSync.setPlaying(value); behavior.speaking = Boolean(value); if (value) behavior.setState('speaking'); },
    feedAudioLevel(rms, timestamp) { lipSync.feed(rms, timestamp); },
    playMotion(group, index = 0) { return renderer.playMotion(group, index); },
    express(name) { const accepted = renderer.playExpression(name); if (accepted) action = name; return accepted; },
    setPointerGaze(x, y) { behavior.setPointerGaze(x, y); },
    triggerGesture(name, options) {
      const accepted = behavior.triggerGesture(name, options);
      if (accepted) action = String(name).toLowerCase();
      return accepted;
    },
    sleep() { behavior.setState('sleeping'); window.pet?.activate?.('lie'); },
    wake() { behavior.setState('idle'); window.pet?.activate?.('sit'); },
    show, hide,
    hitTest(x, y) { return renderer.hitTest(x, y); },
    getStatus() { return { ...renderer.getStatus(), pose, active, state: behavior.state, emotion: behavior.emotion,
      action, gesture: behavior.gesture?.name || null, speaking: lipSync.playing, rms: lipSync.rms,
      mouthOpen: lipSync.value, fps: Math.round(fps), modelPath, modelError, parameters: parameters.output }; },
    destroy() { hide(); renderer.destroy(); }
  };
  window.avatar = avatar;
  const requested = new URLSearchParams(location.search).get('model') || 'assets/live2d/rattan/rattan.model3.json';
  setTimeout(() => loadModel(requested), 100);
})();
