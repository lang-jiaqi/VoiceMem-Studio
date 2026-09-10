"""Studio speech implementation."""
import asyncio
from studio.core.utils.dialogue.component import backchannel_policy_summary
from studio.web import transport as utils

class Speech:
    def _backchannel_tts(self):
        if self.MODE == "realtime":
            rt = str(getattr(utils, "RT_VOICE", "") or "")
            if rt not in self._TTS_SHARED_VOICES:
                print(f"[backchannel] Realtime 音色 {rt!r} 在 TTS API 里没有对应的，"
                      f"合出来会是另一个人的声音 → 这条路关闭附和。"
                      f"想开就把 OPENAI_REALTIME_VOICE 换成 "
                      f"{'/'.join(sorted(self._TTS_SHARED_VOICES))} 之一。", flush=True)
                return None
            from studio.core.utils.tts.providers import OpenAITTS
            return OpenAITTS(voice=rt)
        tts = self.vm.utils.get("tts")

        if getattr(tts, "slow", False) or type(tts).__name__ in self._SLOW_TTS:
            print(f"[backchannel] {type(tts).__name__} 合成比实时还慢，预合成 150 段会把"
                  f"正文回复堵死 → 这条路关闭附和。想要附和请用 支持实时合成的 TTS 组件。",
                  flush=True)
            return None
        return tts

    def _backchannel_voice(self):
        from studio.core.utils.turn_taking.initialize import BackchannelVoice
        if self._BC_VOICE["obj"] is None:
            try:
                tts = self._backchannel_tts()
                if tts is None:
                    self._BC_VOICE["obj"] = False
                    return None
                self._BC_VOICE["obj"] = BackchannelVoice(
                    tts, lang=self.space_language(self.ACTIVE_SPACE),

                    voice_id=str(getattr(tts, "voice", "") or type(tts).__name__))
            except Exception as e:
                print(f"[backchannel] 拿不到 TTS，附和关闭：{type(e).__name__}: {e}", flush=True)
                self._BC_VOICE["obj"] = False
                return None
        v = self._BC_VOICE["obj"]
        if v is False:
            return None
        if self._BC_VOICE["task"] is None and not v.primed:

            def _done(t):
                self._BC_VOICE["task"] = None
                try:
                    t.result()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[backchannel] 预合成失败：{type(e).__name__}: {e}", flush=True)
            # The demo ships the reviewed bank in voice/backchannel/. Never turn a
            # colleague's first launch into a synthesis job: missing clips simply
            # mean that opportunity is silent.
            self._BC_VOICE["task"] = asyncio.create_task(v.prime(cache_only=True))
            self._BC_VOICE["task"].add_done_callback(_done)
        return v if v.ready else None

    def _print_backchannel_status(self) -> None:
        from studio.core.utils.turn_taking.initialize import backchannel as _bc
        if not _bc.emitting():
            print("[backchannel] 关闭。打开：--backchannel", flush=True)
            return
        lang = self.space_language(self.ACTIVE_SPACE)
        words = "/".join(sorted({t for g in _bc._TOKENS[lang].values() for t in g})[:5])
        tts = None
        try:
            tts = self._backchannel_tts()
        except Exception as e:
            print(f"[backchannel] 拿不到 TTS：{type(e).__name__}: {e}", flush=True)
        if tts is None:
            return
        p = backchannel_policy_summary()
        print(f"[backchannel] 开启 · 空间「{self.ACTIVE_SPACE}」语言={lang} → 会说：{words} …",
              flush=True)
        print(f"[backchannel] 音色={getattr(tts, 'voice', '?')}（跟正文同一个）· {p}",
              flush=True)
        print("[backchannel] 想看每次判定：--verbose；"
              "控制参数和 system prompt：studio/harness/turn_taking/policy.py（3秒内不重复附和）", flush=True)

    def _eot(self):
        if not self.ARGS.eot:
            return None
        if self._EOT['obj'] is None:
            from studio.core.utils.eot.initialize import create
            self._EOT['obj'] = create()
        return self._EOT['obj']
