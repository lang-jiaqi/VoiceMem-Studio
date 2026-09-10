"""In-memory context for conversation turns not covered by persistent memory."""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass

@dataclass(frozen=True)
class SessionTurn:
    turn_id: str
    user_text: str
    assistant_text: str
    interrupted: bool = False

_RECENT_KEEP = 6

class SessionBuffer:
    """Store uncommitted turns per Memory Space.

    A turn remains available to the reply model until memory ingestion reports
    that it created persistent memory. The lock is required because ingestion
    completion runs in a background thread.
    """

    def __init__(self, text_limit: int = 200):
        self.text_limit = text_limit
        self._contexts: dict[tuple[str, str], list[SessionTurn]] = {}

        self._recent: dict[tuple[str, str], list[SessionTurn]] = {}
        self._turn_context: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    def add(self, session_id: str, space: str, user_text: str, assistant_text: str,
            interrupted: bool = False) -> str:
        turn_id = uuid.uuid4().hex
        turn = SessionTurn(
            turn_id=turn_id,
            user_text=(user_text or "").strip()[:self.text_limit],
            assistant_text=(assistant_text or "").strip()[:self.text_limit],
            interrupted=bool(interrupted),
        )
        if not turn.user_text and not turn.assistant_text:
            return ""
        with self._lock:
            key = (session_id, space)
            recent = self._recent.setdefault(key, [])
            recent.append(turn)
            del recent[:-_RECENT_KEEP]
            context = (session_id, space)
            self._contexts.setdefault(context, []).append(turn)
            self._turn_context[turn_id] = context
        return turn_id

    def mark_complete(self, turn_id: str, committed: bool) -> None:
        """Remove a turn only after durable memory was actually created."""
        if not turn_id:
            return
        with self._lock:
            context = self._turn_context.pop(turn_id, None)
            if context is None or not committed:
                return
            turns = self._contexts.get(context, [])
            self._contexts[context] = [turn for turn in turns if turn.turn_id != turn_id]

    def turns(self, session_id: str, space: str) -> list[SessionTurn]:
        with self._lock:
            return list(self._contexts.get((session_id, space), []))

    def render(self, session_id: str, space: str, language: str = "zh") -> str:
        turns = self.turns(session_id, space)
        if not turns:
            return ""
        if language == "en":
            lines = ["Conversation in this session not yet stored in long-term memory (latest last):"]
            user_label, assistant_label = "User", "Assistant"
        else:
            lines = ["本次会话中尚未写入长期记忆的对话（最后一条离现在最近）："]
            user_label, assistant_label = "用户", "你"
        for turn in turns:
            lines.append(f"{user_label}: {turn.user_text}")
            label = assistant_label
            if turn.interrupted:
                label += " (interrupted)" if language == "en" else "（被打断）"
            lines.append(f"{label}: {turn.assistant_text}")
        return "\n".join(lines)

    def recent(self, session_id: str, space: str, n: int) -> list["SessionTurn"]:
        """Return bounded recent turns that have not been persisted."""
        with self._lock:
            return list(self._recent.get((session_id, space), []))[-n:]

    def messages(self, session_id: str, space: str, window: int = 0) -> list[dict]:
        """Return provider-neutral dialogue messages for this session and space."""
        turns = (self.recent(session_id, space, window) if window
                 else self.turns(session_id, space))
        out: list[dict] = []
        for turn in turns:
            if turn.user_text:
                out.append({"role": "user", "content": turn.user_text})
            if turn.assistant_text:
                text = turn.assistant_text
                if turn.interrupted:
                    text += "（被用户打断）"
                out.append({"role": "assistant", "content": text})
        return out

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            contexts = [context for context in self._contexts if context[0] == session_id]
            for context in contexts:
                for turn in self._contexts.pop(context, []):
                    self._turn_context.pop(turn.turn_id, None)
