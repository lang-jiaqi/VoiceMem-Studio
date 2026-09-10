"""Studio context implementation."""
from studio.core.utils.dialogue.component import CONTEXT, system_prompt
from voicemem import gate
from voicemem import persona
from studio.core.utils.speaking_style.component import content_emotion_note

class Context:
    def _tone_note(self, emotion: str) -> str:
        return self._by_lang(self._TONE).get((emotion or "").strip(), "")

    def _speak_instruction(self, emotion: str) -> str:
        base = self._speak_base_env or self._by_lang(self._SPEAK_BASE)
        tone = self._tone_note(emotion)
        return f"{base}{tone}" if tone else base

    def _by_lang(self, d: dict, lang: str = "") -> str:
        return d if isinstance(d, str) else d.get("zh", d)

    def _rt_persona(self, lang: str = "") -> str:
        lang = lang or self.SPACE_LANG
        return system_prompt(lang, tagged=self.MODE != "realtime", reply=getattr(self, "REPLY", None))

    def _history_block(self, session_id: str, space: str) -> str:
        from voicemem.lang import is_zh
        return self._SESSION_CONTEXT.render(session_id, space, "zh" if is_zh() else "en")

    def _push_history(self, session_id: str, space: str, user_text: str, reply_text: str,
                      interrupted: bool = False) -> str:
        return self._SESSION_CONTEXT.add(
            session_id, space, user_text, reply_text, interrupted=interrupted)

    def _finish_history_turn(self, turn_id: str, result: dict) -> None:
        committed = bool((result or {}).get("persistent_memory_created"))
        self._SESSION_CONTEXT.mark_complete(turn_id, committed)
        if self.BARGE_DEBUG:
            state = "已进入长期记忆，移出 SessionBuffer" if committed else "未入库，保留短期上下文"
            print(f"[context] turn={turn_id[:8] or '-'} {state}", flush=True)

    def _realtime_instructions(self, memory_context: str, stranger: bool = False,
                               replay: bool = False, emotion: str = "",
                               text: str = "", context_session: str = "",
                               context_space: str = "") -> str:
        if stranger:
            out = f"{self._rt_persona()}\n\n{self._by_lang(CONTEXT['stranger'])}"
            return out
        parts = [self._rt_persona()]
        if memory_context:
            parts.append(memory_context)
        else:
            parts.append(self._by_lang(CONTEXT["no_memory"]))
        session_context = self._history_block(
            context_session, context_space or self.ACTIVE_SPACE)
        if session_context:
            parts.append(session_context)
        tone = self._tone_note(emotion)
        if tone:
            parts.append(self._by_lang(self._STATE_LABEL) + tone)
        content_emotion = content_emotion_note(emotion, self.SPACE_LANG)
        if content_emotion:
            parts.append(content_emotion)
        if replay:
            parts.append(self._by_lang(self._REPLAY_NOTE))
        elif self._wants_sound(text):
            parts.append(self._by_lang(self._NO_REPLAY_NOTE))
        return "\n\n".join(parts)

    def build_reply_context(self, memory_context: str, *, stranger: bool = False,
                            route=None, replay: str = "", text: str = "",
                            emotion: str = "", continuation: bool = False) -> str:
        """Compose history, eligible memory, and dialogue directives for a reply."""
        ctx = self._by_lang(CONTEXT["stranger"]) if stranger else (memory_context or "")
        if not stranger and gate.needs_memory(route) and not ctx.strip():
            ctx = self._by_lang(CONTEXT["no_memory"])
        note = (self._by_lang(self._REPLAY_NOTE) if replay
                else (self._by_lang(self._NO_REPLAY_NOTE) if self._wants_sound(text) else ""))
        if note:
            ctx = f"{ctx}\n\n{note}" if ctx else note
        emotion_note = content_emotion_note(emotion, self.SPACE_LANG)
        if emotion_note:
            ctx = f"{ctx}\n\n{emotion_note}" if ctx else emotion_note
        if continuation:
            followup_note = self._by_lang(CONTEXT["unfinished_followup"])
            ctx = f"{ctx}\n\n{followup_note}" if ctx else followup_note
        return ctx
