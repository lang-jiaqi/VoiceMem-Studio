/* Map VoiceMem events to the public avatar API. Mouth motion only follows actual playback RMS. */
(() => {
  const url = new URLSearchParams(location.search).get('ws'); if (!url) return;
  const seen = new Set();
  let socket, retries = 0, idleTimer, output = '', session = `connection-${Date.now()}`;
  let speaking = false, lipActive = false;
  let lastSadAt = -Infinity;
  const remember = message => {
    const key = [message.session_id || session, message.output_id || '-', message.type,
      message.event_id || '-', message.state || '-', message.rendered_samples ?? '-',
      message.active ?? '-', message.emotion || '-'].join(':');
    if (seen.has(key)) return false; seen.add(key);
    if (seen.size > 256) seen.delete(seen.values().next().value); return true;
  };
  function armSleep() { clearTimeout(idleTimer); idleTimer = setTimeout(() => window.avatar.sleep(), 20000); }
  function stopSpeaking(next = 'idle') {
    speaking = false; lipActive = false; output = ''; window.avatar.setSpeaking(false); window.avatar.setState(next); armSleep();
  }
  function wake(state = 'idle') { clearTimeout(idleTimer); window.avatar.wake(); window.avatar.setState(state); }
  function handle(message) {
    if (!message || typeof message !== 'object' || !remember(message)) return;
    if (message.session_id) session = String(message.session_id);
    if (message.type === 'conversation_started') { wake('listening'); armSleep(); return; }
    if (message.type === 'conversation_ended') { stopSpeaking(); return; }
    if (message.type === 'user_voice') {
      if (message.active === true) wake('listening');
      else { window.avatar.setState(speaking ? 'speaking' : 'thinking'); armSleep(); }
      return;
    }
    if (message.type === 'backchannel') { wake('listening'); window.avatar.triggerGesture('backchannel'); return; }
    if ((message.type === 'memory_hits' || message.type === 'tag_update') &&
        ['悲伤', 'sad'].includes(String(message.emotion || '').trim().toLowerCase())) {
      if (performance.now() - lastSadAt > 6000) { lastSadAt = performance.now(); wake('listening'); window.avatar.express('cry'); }
      return;
    }
    if (message.type === 'answer_interrupt') { stopSpeaking('interrupted'); return; }
    if (message.type === 'avatar_audio_level') {
      if (message.output_id && (!output || output === message.output_id)) {
        if (!speaking) wake('speaking');
        output = message.output_id; speaking = true;
        if (!lipActive) { window.avatar.setSpeaking(true); lipActive = true; }
        window.avatar.feedAudioLevel(message.rms, performance.now());
      }
      return;
    }
    if (message.type === 'error') { stopSpeaking('error'); window.avatar.setEmotion('concerned', .8, 3000); return; }
    if (message.type !== 'playback_checkpoint' || !message.output_id) return;
    if (message.state === 'playing') {
      if (!speaking || output !== message.output_id) wake('speaking');
      output = message.output_id; speaking = true;
      if (!lipActive) { window.avatar.setSpeaking(true); lipActive = true; }
      return;
    }
    if (message.output_id !== output) return;
    if (message.state === 'paused' || message.state === 'stalled') { lipActive = false; window.avatar.setSpeaking(false); return; }
    if (message.state === 'interrupted') { stopSpeaking('interrupted'); return; }
    if (message.state === 'drained') {
      stopSpeaking();
    }
  }
  function schedule() { setTimeout(connect, Math.min(5000, 250 * 2 ** retries++)); }
  function connect() {
    try { socket = new WebSocket(url); } catch { schedule(); return; }
    socket.onopen = () => { retries = 0; session = `connection-${Date.now()}`; };
    socket.onmessage = incoming => { try { handle(JSON.parse(incoming.data)); } catch {} };
    socket.onerror = () => { try { socket.close(); } catch {} };
    socket.onclose = () => { stopSpeaking(); schedule(); };
  }
  connect();
})();
