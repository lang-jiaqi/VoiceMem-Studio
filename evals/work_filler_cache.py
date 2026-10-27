"""Prepare local waiting-speech banks and time reuse without playback or APIs.

This intentionally writes synthesized speech to the ignored waiting-speech cache.
Run on the native platform supported by the selected local speech backend.
"""
import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time


async def main(args):
    from studio.core.utils.component_config.component import selection
    if selection('speech')['provider'] != 'recommended':
        raise ValueError('This check requires local recommended speech; API adapters are not invoked')
    from studio.core.utils.context.component import Context
    from studio.core.utils.self_harness.component import SelfHarnessState
    from studio.core.utils.tts.initialize import create
    from studio.core.utils.turn_taking.filler_voice import WorkFillerVoice, instruction
    from studio.prompt_config import tts_prompts
    agent = Context()
    agent._SPEAK_BASE = tts_prompts()['base']
    agent._TONE = tts_prompts()['fallback_by_user_emotion']
    agent._speak_base_env = ''
    snapshot = SelfHarnessState().snapshot()
    tts, calls = create(), []
    original = tts.stream

    def stream(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    tts.stream = stream
    report = {'scope': 'local_waiting_speech_cache_no_playback_no_api', 'banks': {}}
    for language in ('zh', 'en'):
        style = instruction(agent, language, snapshot)
        bank = WorkFillerVoice(tts, language, style)
        started, previous_calls = time.perf_counter(), len(calls)
        await bank.prime()
        prepared_ms = (time.perf_counter() - started) * 1000
        durations = [len(pcm) / 48000 for pcm in bank.clips.values()]
        reloaded = WorkFillerVoice(tts, language, style)
        reload_started, reload_calls = time.perf_counter(), len(calls)
        await reloaded.prime()
        reload_ms = (time.perf_counter() - reload_started) * 1000
        hit_times, recent = [], []
        for _ in range(1000):
            started = time.perf_counter()
            text, _ = reloaded.choose(recent)
            hit_times.append((time.perf_counter() - started) * 1000)
            if text in recent:
                raise AssertionError('Repeated an emitted phrase inside the last-three window')
            recent = (recent + [text])[-3:]
        report['banks'][language] = {
            'phrases': len(bank.clips), 'prepare_ms': prepared_ms,
            'new_syntheses': reload_calls - previous_calls,
            'audio_durations_s': durations, 'disk_reload_ms': reload_ms,
            'reload_syntheses': len(calls) - reload_calls,
            'memory_hit_median_ms': statistics.median(hit_times),
            'memory_hit_p95_ms': sorted(hit_times)[int(.95 * (len(hit_times) - 1))],
        }
        print(json.dumps({language: report['banks'][language]}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
