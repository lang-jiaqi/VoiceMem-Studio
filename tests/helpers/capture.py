"""Synthetic capture frames and isolated state for Studio turn-taking tests."""
import asyncio
import base64
import dataclasses
import json
import re
import time
import types
from contextlib import aclosing
from unittest.mock import patch

import numpy as np
from studio.core.utils.echo_guard.component import UtteranceGuard
from studio.core.utils.reply_modes.initialize import DIRECT, MEMORY
from studio.core.utils.turn_taking.initialize import Backchannel, TurnTakingStateMachine
from voicemem import gate
from voicemem.stream import StreamState
from web.harness import PauseGate, backchannel_policy, is_unfinished
from tests.helpers.studio import studio_tree, execute


def anticipate_namespace():
    tree = studio_tree()
    names = {"anticipate", "_capture_turns", "Pending", "_is_echo", "_is_backchannel", "_barge_text",
             "_is_explicit_interrupt", "_has_barge_content", "_has_strong_final_barge"}
    ns = dict(asyncio=asyncio, base64=base64, json=json, time=time, aclosing=aclosing,
              open_stream=lambda memory, **kw: memory.stream(**kw), PauseGate=PauseGate, backchannel_policy=backchannel_policy, is_unfinished=is_unfinished,
              Backchannel=Backchannel, TurnTakingStateMachine=TurnTakingStateMachine,
              DIRECT=DIRECT, MEMORY=MEMORY,
              dataclass=dataclasses.dataclass, gate=gate, UtteranceGuard=UtteranceGuard,
              BACKCHANNEL_ON=True, ECHO_WINDOW=300, ECHO_RATIO=.6, ECHO_FUZZY_MIN=4,
              _bc_norm=gate.norm, _FILLER_PREFIX=re.compile(r"^[嗯呃啊哦噢喔欸诶唉哈哼]+"),
              _INTERRUPT_PREFIXES=("停", "等一下", "stop", "wait"), BARGE_MIN_CHARS=2,
              SPEC_MIN_CHARS=6, GAMBLE_S=.2, CONFIRM_S=.2, MIC_RATE=24000,
              BC_ECHO_WINDOW_S=3, BC_QUIET_RATIO=.25, BC_AFTER_EARLY_S=1,
              EARLY_EOT=.5, CONFIRM_READY_S=.1,
              CANDIDATE_MIN_SPEECH_S=.04, BARGE_STABLE_UPDATES=2,
              BARGE_REJECT_SILENCE_MS=200, BARGE_CANDIDATE_TIMEOUT_MS=1200,
              BARGE_DEBUG=False, SPEAKER_GATE=False, SPEAKER_DEBUG=False,
              STRANGER_MIN_TURNS=1, ACTIVE_SPACE="test", _eot=lambda: None,
              space_language=lambda _: "zh", _replaying_now=lambda: False,
              _replay_id=lambda *a: "", save_turn_audio=lambda *a: "")
    execute([n for n in tree.body if getattr(n, "name", "") in names], ns)
    return ns


def state(text="", *, final=False):
    turn = types.SimpleNamespace(text=text, raw_text=text, result=object(),
                                 memory_context="", route="shallow") if final else None
    return StreamState("turn_over" if final else "<speak>", text, None, turn,
                       spoke=not final, speech_end=time.monotonic())


class CaptureFixture:
    async def run_frames(self, frames, on_early=None, on_early_cancel=None):
        ns, sent, yielded, interrupted = anticipate_namespace(), [], [], []
        playback = {"busy": False, "text": ""}
        frames = iter(frames)
        current = None
        class Sock:
            async def receive(self):
                nonlocal current
                current = next(frames, None)
                if current is None:
                    return {"type": "websocket.disconnect"}
                playback.update(busy=current[0], text=current[1])
                return {"bytes": np.zeros(960, np.int16).tobytes()}
            async def send_json(self, message):
                sent.append(message)
        async def feed(_):
            if len(current) > 3:  # playback can drain inside final ASR's await
                playback.update(busy=current[3], text="")
            return current[2]
        async def stop():
            interrupted.append(True)
        def refine_current_snapshot():
            return asyncio.create_task(asyncio.sleep(0, result=current[2].text))
        stream = types.SimpleNamespace(feed=feed, confirm_s=.2,
                                       refine_current_snapshot=refine_current_snapshot)
        ns["vm"] = types.SimpleNamespace(stream=lambda **kw: stream)
        with patch("studio.core.utils.turn_taking.backchannel.emitting", return_value=False), \
             patch("studio.core.utils.turn_taking.backchannel.Backchannel.offer", return_value=None):
            async for pending in ns["anticipate"](
                    Sock(), is_busy=lambda: playback["busy"],
                    said=lambda: playback["text"], on_speech=stop,
                    on_early=on_early, on_early_cancel=on_early_cancel):
                yielded.append(pending)
        return sent, yielded, interrupted

