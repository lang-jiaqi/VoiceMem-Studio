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

    Recent history retains persisted turns within a character budget; pending
    context is removed when ingestion confirms durable memory. The lock is
    required because ingestion completion runs in a background thread.
    """

    def __init__(self, text_limit: int = 4000, context_limit: int = 12000):
        if text_limit <= 0 or context_limit < 2 * text_limit:
            raise ValueError("Context budget must cover one bounded user/assistant pair")
        self.text_limit = text_limit
        self.context_limit = context_limit
        self._contexts: dict[tuple[str, str], list[SessionTurn]] = {}

        self._recent: dict[tuple[str, str], list[SessionTurn]] = {}
        self._turn_context: dict[str, tuple[str, str]] = {}
        self._lock = threading.RLock()

    def _limit_text(self, value: str) -> str:
        value = (value or '').strip()
        if len(value) <= self.text_limit:
            return value
        if self.text_limit < 5:
            return value[:self.text_limit]
        available = self.text_limit - 3
        head = available // 2
        return value[:head] + '\n…\n' + value[-(available - head):]

    def _trim(self, turns, count=None):
        removed = []
        total = sum(len(turn.user_text) + len(turn.assistant_text) for turn in turns)
        while len(turns) > 1 and ((count is not None and len(turns) > count)
                                  or total > self.context_limit):
            turn = turns.pop(0)
            total -= len(turn.user_text) + len(turn.assistant_text)
            removed.append(turn)
        return removed

    def add(self, session_id: str, space: str, user_text: str, assistant_text: str,
            interrupted: bool = False) -> str:
        turn_id = uuid.uuid4().hex
        turn = SessionTurn(
            turn_id=turn_id,
            user_text=self._limit_text(user_text),
            assistant_text=self._limit_text(assistant_text),
            interrupted=bool(interrupted),
        )
        if not turn.user_text and not turn.assistant_text:
            return ""
        with self._lock:
            key = (session_id, space)
            recent = self._recent.setdefault(key, [])
            recent.append(turn)
            self._trim(recent, _RECENT_KEEP)
            context = (session_id, space)
            pending = self._contexts.setdefault(context, [])
            pending.append(turn)
            for removed in self._trim(pending):
                self._turn_context.pop(removed.turn_id, None)
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
        """Return bounded recent turns, including those already persisted."""
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
                out.append({"role": "assistant", "content": turn.assistant_text})
        return out

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            contexts = [context for context in self._contexts if context[0] == session_id]
            for context in contexts:
                for turn in self._contexts.pop(context, []):
                    self._turn_context.pop(turn.turn_id, None)
            recent = [context for context in self._recent if context[0] == session_id]
            for context in recent:
                self._recent.pop(context, None)
