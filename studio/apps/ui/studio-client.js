/* Same-origin Studio transport. Each page owns one cancellable conversation. */
(() => {
  'use strict';
  const RATE = 24000;
  const MAX_REPLAY_RECORDS = 32;
  const MAX_REPLAY_SAMPLES = RATE * 60 * 20;
  const SHORT_CAPTION_LENGTH = 80;
  const MAX_CAPTION_PHRASE = 64;
  function captionEnd(caption) {
    const {text, shown, limit, textDone, drained} = caption;
    if (drained || (textDone && text.length <= SHORT_CAPTION_LENGTH)) return text.length;
    let end = shown;
    // Preview one complete phrase; heard progress remains owned by the server.
    while (end <= limit && end < text.length) {
      const start = end;
      let next = start, space = start;
      for (let i = start; i < text.length; i++) {
        const char = text[i], count = i - start + 1;
        if (/\s/u.test(char)) space = i + 1;
        const following = text[i + 1] || '';
        const numeric = /\d/u.test(text[i - 1] || '')
          && (/\d/u.test(following) || (!following && !textDone));
        const sentence = /[。！？!?；;\n]/u.test(char)
          || (char === '.' && !numeric && (!following || /[\s"'”’)]/u.test(following)));
        const clause = /[，,：:]/u.test(char) && !numeric && count >= 8;
        if (sentence || clause) {
          next = i + 1;
          while (next < text.length && /[\s。！？!?；;”’」』）》)\]]/u.test(text[next])) next++;
          break;
        }
        if (count >= MAX_CAPTION_PHRASE) {
          next = space > start + MAX_CAPTION_PHRASE / 2 ? space : i + 1;
          break;
        }
      }
      if (next === start) next = textDone ? text.length : start;
      if (next === start) break;
      end = next;
    }
    return end;
  }
  function resample(input, from, to) {
    if (from === to) return input;
    const ratio = from / to;
    const output = new Float32Array(Math.max(1, Math.round(input.length / ratio)));
    for (let i = 0; i < output.length; i++) {
      const position = i * ratio, a = Math.min(Math.floor(position), input.length - 1);
      const b = Math.min(a + 1, input.length - 1), t = position - a;
      output[i] = input[a] * (1 - t) + input[b] * t;
    }
    return output;
  }
  function create({ onEvent = () => {}, onState = () => {}, onPhase = () => {}, onAudioLevel = () => {} } = {}) {
    let run = null;
    let replayRun = null;
    let replaySamples = 0;
    let promptSequence = 0;
    const recordings = new Map();
    const active = owner => !!owner && run === owner && !owner.closed;
    const emit = (owner, event) => { if (active(owner)) onEvent(event); };
    const sendJSON = (owner, message) => {
      if (active(owner) && owner.socket?.readyState === WebSocket.OPEN)
        owner.socket.send(JSON.stringify(message));
    };
    function stopMic(owner) {
      const mic = owner?.mic;
      if (!mic) return;
      owner.mic = null;
      mic.stream?.getTracks().forEach(track => track.stop());
      if (mic.node) {
        mic.node.port.onmessage = null;
        mic.node.port.close();
        mic.node.disconnect();
        try { owner.bus.disconnect(mic.node); } catch {}
      }
      mic.source?.disconnect();
      onState(false);
    }
    function deleteRecording(id) {
      const recording = recordings.get(id);
      if (!recording) return;
      replaySamples -= recording.frames;
      recordings.delete(id);
    }
    function pruneRecordings(protectedId = '') {
      while (recordings.size > MAX_REPLAY_RECORDS || replaySamples > MAX_REPLAY_SAMPLES) {
        const candidate = [...recordings.values()].find(recording =>
          recording.id !== protectedId && recording.state !== 'recording');
        if (!candidate) break;
        deleteRecording(candidate.id);
      }
    }
    function beginRecording(id, sampleRate) {
      id = String(id || '');
      if (!id) return;
      deleteRecording(id);
      recordings.set(id, {
        id, sampleRate: Math.max(1, Number(sampleRate) || RATE), chunks: [], frames: 0,
        state: 'recording',
      });
      pruneRecordings(id);
    }
    function appendRecording(id, pcm) {
      const recording = recordings.get(id);
      if (!recording || recording.state !== 'recording' || !pcm.length) return;
      const copy = pcm.slice();
      recording.chunks.push(copy); recording.frames += copy.length; replaySamples += copy.length;
      pruneRecordings(id);
      if (replaySamples > MAX_REPLAY_SAMPLES && recordings.size === 1) deleteRecording(id);
    }
    function finishRecording(id, renderedFrames, interrupted = false) {
      const recording = recordings.get(id);
      if (!recording || recording.state !== 'recording') return;
      const reported = Math.max(0, Math.round(Number(renderedFrames) || 0));
      const keep = interrupted ? Math.min(recording.frames, reported)
        : Math.min(recording.frames, reported || recording.frames);
      if (!keep) { deleteRecording(id); return; }
      if (keep < recording.frames) {
        const chunks = [];
        let remaining = keep;
        for (const chunk of recording.chunks) {
          if (!remaining) break;
          const count = Math.min(remaining, chunk.length);
          chunks.push(count === chunk.length ? chunk : chunk.slice(0, count));
          remaining -= count;
        }
        replaySamples -= recording.frames - keep;
        recording.chunks = chunks; recording.frames = keep;
      }
      recording.state = 'ready';
      recordings.delete(id); recordings.set(id, recording);
      pruneRecordings();
    }
    function stopReplay() {
      const current = replayRun;
      if (!current) return;
      replayRun = null;
      current.source && (current.source.onended = null);
      try { current.source?.stop(); } catch {}
      current.source?.disconnect();
      if (current.ownsContext) void current.context.close().catch(() => {});
      if (current.started) current.onState(false);
    }
    function finishReplay(current) {
      if (replayRun !== current) return;
      replayRun = null;
      current.source.onended = null; current.source.disconnect();
      if (current.ownsContext) void current.context.close().catch(() => {});
      if (current.started) current.onState(false);
    }
    async function replay(outputId, onReplayState = () => {}) {
      const id = String(outputId || '');
      if (replayRun?.id === id) { stopReplay(); return false; }
      stopReplay();
      const recording = recordings.get(id);
      if (!recording || recording.state !== 'ready' || !recording.frames) {
        VMUI.notify('这条回复的原始语音暂不可用。');
        return false;
      }
      const owner = active(run) && run.context && run.bus ? run : null;
      let context;
      try {
        context = owner?.context || new AudioContext({ sampleRate: RATE, latencyHint: 'playback' });
        const current = replayRun = {
          id, context, ownsContext: !owner, source: null, started: false, onState: onReplayState,
        };
        await context.resume();
        if (replayRun !== current) return false;
        const buffer = context.createBuffer(1, recording.frames, recording.sampleRate);
        const samples = buffer.getChannelData(0);
        let offset = 0;
        for (const chunk of recording.chunks) {
          for (let i = 0; i < chunk.length; i++) samples[offset++] = chunk[i] / 32768;
        }
        const source = current.source = context.createBufferSource();
        source.buffer = buffer; source.connect(owner?.bus || context.destination);
        source.onended = () => finishReplay(current);
        source.start(); current.started = true; onReplayState(true);
        return true;
      } catch (error) {
        if (replayRun?.id === id) stopReplay();
        else if (context && context !== run?.context) void context.close().catch(() => {});
        VMUI.notify(error.message || '原始语音播放失败。');
        return false;
      }
    }
    function end() {
      stopReplay();
      const owner = run;
      if (!owner) return;
      if (recordings.get(owner.output)?.state === 'recording') deleteRecording(owner.output);
      stopMic(owner);
      owner.replyEvents = []; owner.replyStarted = false; owner.caption = null;
      owner.closed = true;
      run = null;
      clearTimeout(owner.timeout);
      clearTimeout(owner.thinking);
      for (const request of owner.promptRequests?.values() || []) {
        clearTimeout(request.timeout); request.reject(new Error('会话已结束'));
      }
      owner.promptRequests?.clear();
      owner.reject?.(new Error('会话已结束'));
      owner.socket?.close();
      owner.clips.forEach(source => { source.onended = null; try { source.stop(); } catch {} });
      if (owner.player) { owner.player.port.onmessage = null; owner.player.port.close(); owner.player.disconnect(); }
      owner.bus?.disconnect();
      void owner.context?.close().catch(() => {});
      onPhase('idle');
      onEvent({ type: 'disconnected' });
    }
    function fail(owner, error) {
      if (!active(owner)) return;
      end();
      const message = error.message || 'Studio 连接失败，请检查后端是否已启动。';
      VMUI.notify(message);
      window.studioModelServices?.reportConversationError?.(message);
    }
    function phase(owner, value) {
      clearTimeout(owner.thinking);
      if (!active(owner)) return;
      onPhase(value);
      if (value === 'short-thinking') owner.thinking = setTimeout(() => {
        if (active(owner)) onPhase('long-thinking');
      }, 1500);
    }
    function flushReply(owner) {
      if (!owner.replyStarted || owner.paused) return;
      const output = owner.output, caption = owner.caption;
      if (caption) {
        const end = captionEnd(caption);
        if (end > caption.shown) {
          const text = caption.text.slice(caption.shown,end).join('');
          caption.shown = end;
          emit(owner,{type:'answer_delta',output_id:output,text});
          if (!active(owner) || owner.output !== output) return;
        }
        if (caption.drained && caption.done) {
          const done = caption.done; caption.done = null;
          emit(owner,done);
          if (!active(owner) || owner.output !== output) return;
        }
      }
      const events = owner.replyEvents;
      owner.replyEvents = [];
      for (const event of events) {
        if (!active(owner) || owner.output !== output) return;
        emit(owner, event);
      }
    }
    function audio(owner, buffer) {
      if (!owner.output) return;
      const pcm = new Int16Array(buffer), native = new Float32Array(pcm.length);
      appendRecording(owner.output, pcm);
      for (let i = 0; i < pcm.length; i++) native[i] = pcm[i] / 32768;
      const samples = resample(native, owner.rate, owner.context.sampleRate);
      owner.player.port.postMessage({ type: 'audio', samples, sourceFrames: native.length }, [samples.buffer]);
    }
    function backchannel(owner, message) {
      if (!message.pcm) return;
      const bytes = atob(message.pcm), count = bytes.length >> 1;
      if (!count) return;
      const buffer = owner.context.createBuffer(1, count, Number(message.sample_rate) || RATE);
      const samples = buffer.getChannelData(0);
      for (let i = 0; i < count; i++) samples[i] = ((bytes.charCodeAt(i * 2) | bytes.charCodeAt(i * 2 + 1) << 8) << 16 >> 16) / 32768;
      const source = owner.context.createBufferSource(), gain = owner.context.createGain();
      source.buffer = buffer; gain.gain.value = .72;
      source.connect(gain); gain.connect(owner.bus); owner.clips.add(source);
      source.onended = () => {
        owner.clips.delete(source); source.disconnect(); gain.disconnect();
        if (!active(owner)) return;
        if (!owner.clips.size && !owner.paused) owner.player.port.postMessage({ type: 'resume' });
        if (message.filler_id) sendJSON(owner, { type: 'filler_done', filler_id: message.filler_id });
      };
      source.start();
    }
    function handle(owner, message) {
      if (!active(owner)) return;
      if (message.type === 'self_harness_state') {
        const request = owner.promptRequests?.get(message.request_id);
        if (request) {
          clearTimeout(request.timeout); owner.promptRequests.delete(message.request_id);
          if (message.error) request.reject(new Error(message.error)); else request.resolve(message);
        }
        document.dispatchEvent(new CustomEvent('self-harness-state', { detail: message }));
      }
      if (message.type === 'reply_preparing') owner.preparingOutput = message.output_id;
      const preparation = ['reply_preparing', 'memory_hits', 'tag_update'].includes(message.type)
        && message.output_id === owner.preparingOutput;
      if (owner.preparingOutput && ['memory_hits', 'tag_update'].includes(message.type)
          && message.output_id && !preparation) return;
      if (message.output_id && message.type !== 'answer_start' && message.output_id !== owner.output && !preparation) return;
      switch (message.type) {
        case 'session_ready':
          owner.player.port.postMessage({
            type: 'config', prebuffer: message.mode === 'realtime' ? .08 : .16,
            startGrace: message.mode === 'realtime' ? 0 : .04,
          });
          sendJSON(owner, {type:'conversation_language', language:window.VMSettings?.language === 'en' ? 'en' : 'zh'});
          const prompts = window.VMHarnessPrompts?.saved();
          if (prompts && Object.keys(prompts).length) sendJSON(owner, {type:'harness_prompts_update', prompts});
          clearTimeout(owner.timeout); owner.resolve(owner); break;
        case 'user_transcript':
          owner.preparingOutput = '';
          phase(owner, 'short-thinking'); break;
        case 'reply_preparing':
          phase(owner, message.reply_mode === 'memory_cot' ? 'long-thinking' : 'short-thinking'); break;
        case 'answer_start':
          stopReplay();
          owner.output = message.output_id || '';
          owner.preparingOutput = '';
          owner.replyStarted = false;
          owner.replyEvents = [];
          owner.caption = message.caption_sync === 'playback'
            ? {text:[],shown:0,limit:0,textDone:false,done:null,drained:false} : null;
          owner.rate = Number(message.sample_rate) || RATE;
          beginRecording(owner.output, owner.rate);
          owner.paused = false;
          owner.player.port.postMessage({ type: 'start', outputId: owner.output, sampleRate: owner.rate });
          if (owner.clips.size) owner.player.port.postMessage({ type: 'pause' });
          phase(owner, 'short-thinking'); break;
        case 'answer_delta':
          if (!owner.output) return;
          if (owner.caption) {
            for (const char of String(message.text || '')) owner.caption.text.push(char);
            flushReply(owner); return;
          }
          if (!owner.replyStarted || owner.paused) {
            owner.replyEvents.push(message); return;
          }
          break;
        case 'answer_progress':
          if (owner.caption && Number.isSafeInteger(message.text_end) && message.text_end >= 0) {
            owner.caption.limit = Math.max(owner.caption.limit,message.text_end);
            flushReply(owner);
          }
          return;
        case 'answer_text_done':
          if (owner.caption) {
            owner.caption.textDone = true;
            flushReply(owner);
          }
          return;
        case 'answer_pause':
          owner.paused = true; owner.player.port.postMessage({ type: 'pause' }); phase(owner, 'listening'); break;
        case 'answer_resume':
          owner.paused = false;
          if (!owner.clips.size) owner.player.port.postMessage({ type: 'resume' });
          flushReply(owner);
          break;
        case 'answer_done':
          if (!owner.output) return;
          if (owner.caption) { owner.caption.textDone = true; owner.caption.done = message; }
          owner.player.port.postMessage({ type: 'drain' });
          if (owner.caption) { flushReply(owner); return; }
          if (!owner.replyStarted || owner.paused) {
            owner.replyEvents.push(message); return;
          }
          break;
        case 'answer_interrupt':
          owner.replyEvents = []; owner.replyStarted = false; owner.caption = null; owner.paused = false;
          owner.player.port.postMessage({ type: 'clear', reason: 'interrupted' });
          owner.output = ''; phase(owner, owner.mic ? 'listening' : 'idle'); break;
        case 'backchannel':
          backchannel(owner, message); break;
        case 'error':
          owner.replyEvents = []; owner.replyStarted = false; owner.caption = null; owner.paused = false;
          owner.player.port.postMessage({ type: 'clear', reason: 'interrupted' });
          owner.output = ''; phase(owner, owner.mic ? 'listening' : 'idle');
          VMUI.notify(message.message || '服务返回错误，请重试。'); break;
      }
      emit(owner, message);
    }
    async function setup(owner) {
      try {
        const ctx = owner.context;
        await ctx.resume();
        await ctx.audioWorklet.addModule('/pcm-player-worklet.js');
        if (!active(owner)) return;
        owner.bus = ctx.createGain(); owner.bus.connect(ctx.destination);
        owner.player = new AudioWorkletNode(ctx, 'voicemem-pcm-player', { outputChannelCount: [1] });
        owner.player.connect(owner.bus);
        owner.player.port.onmessage = ({ data }) => {
          if (!active(owner)) return;
          if (data.outputId) {
            if (data.type === 'drained') finishRecording(data.outputId, data.renderedSamples);
            else if (data.type === 'interrupted') finishRecording(data.outputId, data.renderedSamples, true);
          }
          const states = { buffer: 'playing', started: 'playing', resumed: 'playing', underflow: 'stalled' };
          if (data.type === 'level' && data.outputId) {
            if (data.outputId === owner.output) onAudioLevel(Number(data.rms || 0));
            sendJSON(owner, {
            type: 'avatar_audio_level', output_id: data.outputId,
            rms: Number(data.rms || 0), peak: Number(data.peak || 0),
            rendered_samples: Number(data.renderedSamples || 0), sample_rate: Number(data.sampleRate || RATE),
            });
          }
          else if (data.outputId) sendJSON(owner, {
            type: 'playback_checkpoint', output_id: data.outputId,
            rendered_samples: Number(data.renderedSamples || 0), sample_rate: Number(data.sampleRate || RATE),
            buffered_samples: Math.round(Number(data.bufferedMs || 0) * Number(data.sampleRate || RATE) / 1000),
            state: states[data.type] || data.type, event: data.type,
          });
          if (data.outputId !== owner.output) return;
          if (data.type === 'started') owner.replyStarted = true;
          if (data.type === 'started') emit(owner, {type:'playback_started', output_id:owner.output});
          if (data.type === 'started' || data.type === 'resumed') {
            flushReply(owner);
            if (owner.replyStarted && !owner.paused) phase(owner, 'speaking');
          }
          if (data.type === 'drained') {
            // A text-only completion has no started checkpoint; keep its events ordered.
            owner.replyStarted = true;
            if (owner.caption) owner.caption.drained = true;
            owner.replyEvents.push({ type: 'playback_done', output_id: owner.output });
            flushReply(owner);
            if (!owner.paused) phase(owner, owner.mic ? 'listening' : 'idle');
            if (owner.memoryClip) {
              const id = owner.memoryClip; owner.memoryClip = '';
              void playMemory(owner, id);
            }
          }
        };
        const url = new URL('/ws', location.href); url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        const socket = owner.socket = new WebSocket(url.href); socket.binaryType = 'arraybuffer';
        socket.onmessage = event => {
          if (!active(owner)) return;
          try {
            if (event.data instanceof ArrayBuffer) audio(owner, event.data);
            else {
              const message = JSON.parse(event.data);
              if (message.type === 'play_memory') owner.memoryClip = message.memory_id;
              handle(owner, message);
            }
          } catch (error) { fail(owner, error); }
        };
        socket.onerror = () => fail(owner, new Error('无法连接 Studio，请先在命令行启动后端。'));
        socket.onclose = () => fail(owner, new Error('Studio 连接已断开，请重新开始对话。'));
      } catch (error) { fail(owner, error); }
    }
    async function playMemory(owner, id) {
      try {
        const response = await fetch(`/api/audio/${encodeURIComponent(id)}`);
        if (!response.ok) throw new Error('记忆录音暂不可用。');
        const buffer = await owner.context.decodeAudioData(await response.arrayBuffer());
        if (!active(owner)) return;
        const source = owner.context.createBufferSource(); source.buffer = buffer;
        source.connect(owner.bus); owner.clips.add(source);
        source.onended = () => { owner.clips.delete(source); source.disconnect(); };
        source.start();
      } catch (error) { if (active(owner)) VMUI.notify(error.message); }
    }
    function connect() {
      if (run) return run.ready;
      const owner = run = { closed: false, clips: new Set(), rate: RATE, output: '', replyStarted: false, replyEvents: [] };
      owner.ready = new Promise((resolve, reject) => { owner.resolve = resolve; owner.reject = reject; });
      // A handler is attached immediately so cancellation never leaves an unhandled rejection.
      owner.ready.catch(() => {});
      owner.timeout = setTimeout(() => fail(owner, new Error('连接超时，请检查 Studio 服务。')), 15000);
      try {
        owner.context = new AudioContext({ sampleRate: RATE, latencyHint: 'interactive' });
        void setup(owner);
      } catch (error) { fail(owner, error); }
      return owner.ready;
    }
    async function start() {
      stopReplay();
      const ready = connect(), owner = run;
      if (!owner || owner.mic) return;
      const mic = owner.mic = {};
      onState(true);
      try {
        await ready;
        if (!active(owner) || owner.mic !== mic) return;
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        if (!active(owner) || owner.mic !== mic) { stream.getTracks().forEach(track => track.stop()); return; }
        mic.stream = stream;
        await owner.context.audioWorklet.addModule('/mic-capture-worklet.js');
        if (!active(owner) || owner.mic !== mic) return;
        mic.source = owner.context.createMediaStreamSource(stream);
        mic.node = new AudioWorkletNode(owner.context, 'voicemem-mic-capture', { numberOfInputs: 2, numberOfOutputs: 1, outputChannelCount: [1] });
        mic.source.connect(mic.node, 0, 0); owner.bus.connect(mic.node, 0, 1); mic.node.connect(owner.context.destination);
        mic.node.port.onmessage = ({ data }) => {
          if (!active(owner) || owner.mic !== mic || data.type !== 'mic' || owner.socket.readyState !== WebSocket.OPEN) return;
          const samples = resample(data.samples, owner.context.sampleRate, RATE), pcm = new Int16Array(samples.length);
          let peak = 0;
          for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); pcm[i] = s < 0 ? s * 32768 : s * 32767; peak = Math.max(peak, Math.abs(s)); }
          window.liquidOrb?.setAudioLevel(peak);
          owner.socket.send(pcm.buffer);
        };
        sendJSON(owner, { type: 'conversation_start' });
        phase(owner, 'listening');
      } catch (error) { if (active(owner)) {
        stopMic(owner);
        const message = `麦克风未启动：${error.message}`;
        VMUI.notify(message); window.studioModelServices?.reportConversationError?.(message);
      } }
    }
    async function send(text) {
      stopReplay();
      const owner = await connect();
      if (!active(owner)) throw new Error('会话已结束');
      sendJSON(owner, { type: 'user_text', text });
      phase(owner, 'short-thinking');
    }
    function pause() {
      if (run) stopMic(run);
    }
    async function getHarness() {
      const owner = await connect();
      if (!active(owner)) throw new Error('会话已结束');
      sendJSON(owner, { type: 'self_harness_get' });
    }
    async function setHarness(update) {
      const owner = await connect();
      if (!active(owner)) throw new Error('会话已结束');
      sendJSON(owner, { type: 'self_harness_update', update });
    }
    async function setHarnessPrompt(name, value) {
      const owner = await connect();
      if (!active(owner)) throw new Error('会话已结束');
      sendJSON(owner, { type: 'harness_prompt_update', name, value });
    }
    async function setHarnessPrompts(prompts, promptLanguage) {
      const owner = await connect();
      if (!active(owner)) throw new Error('会话已结束');
      const request_id = `prompts-${++promptSequence}`;
      owner.promptRequests ||= new Map();
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          owner.promptRequests.delete(request_id); reject(new Error('Prompt 保存超时，请重试。'));
        }, 10000);
        owner.promptRequests.set(request_id, {resolve, reject, timeout});
        sendJSON(owner, {type:'harness_prompts_update', prompts, request_id,
          ...(promptLanguage ? {prompt_language:promptLanguage} : {})});
      });
    }
    async function setBackchannelCurve(values) {
      const owner = await connect();
      if (!active(owner)) throw new Error('会话已结束');
      sendJSON(owner, { type: 'backchannel_curve_update', values });
    }
    function toggle() { if (run?.mic) end(); else void start(); }
    window.addEventListener('pagehide', end);
    document.addEventListener('visibilitychange', () => { if (document.hidden && !window.studioModelServices) end(); });
    document.addEventListener('settings-open', pause);
    let language = window.VMSettings?.language || 'zh-CN';
    document.addEventListener('display-settings-change', () => {
      const next = window.VMSettings?.language || 'zh-CN';
      if (next === language) return;
      language = next;
      if (run) sendJSON(run, {type:'conversation_language', language:next === 'en' ? 'en' : 'zh'});
      fetch('/api/lang', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ lang: next === 'en' ? 'en' : 'zh' }) })
        .then(response => { if (!response.ok) throw new Error('语言切换失败'); })
        .catch(error => VMUI.notify(error.message));
    });
    return { send, start, toggle, pause, getHarness, setHarness, setHarnessPrompt, setHarnessPrompts, setBackchannelCurve, replay, stopReplay, cancel: end, stop: end };
  }
  function applyUser(messages, event, role) {
    const id = event.input_turn_id;
    const finalized = messages.find(message => message.role === role && message.inputIds?.includes(id));
    if (finalized) return finalized;
    let item = messages.find(message => message.role === role &&
      ((id && message.inputId === id) || (event.replace_input_turn_id &&
        (message.inputId === event.replace_input_turn_id || message.inputIds?.includes(event.replace_input_turn_id)))));
    if (!item) item = messages.find(message => message.role === role && message.pending && message.text === event.text);
    if (!item) { item = {role, time:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'})}; messages.push(item); }
    Object.assign(item, {text:event.text, inputId:id, pending:false,
      inputIds:[...new Set([...(item.inputIds || []), id].filter(Boolean))]});
    return item;
  }
  function acceptsPartial(messages, event) {
    return !event.input_turn_id || !messages.some(message => message.inputIds?.includes(event.input_turn_id));
  }
  window.VMStudio = { create, applyUser, acceptsPartial };
})();
