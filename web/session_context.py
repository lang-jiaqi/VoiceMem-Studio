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


#: 给回复模型看的对话历史保留多少轮。滑窗，进没进长期记忆都不影响。
_RECENT_KEEP = int(os.environ.get("VOICEMEM_HISTORY_TURNS", "6"))


class SessionBuffer:
    """Store uncommitted turns per Memory Space.

    A turn remains available to the reply model until memory ingestion reports
    that it created persistent memory. The lock is required because ingestion
    completion runs in a background thread.
    """

    def __init__(self, text_limit: int = 200):
        self.text_limit = text_limit
        self._contexts: dict[tuple[str, str], list[SessionTurn]] = {}
        #: 最近若干轮，**不因入库而移除**——给回复模型看的对话历史。
        #: _contexts 是另一回事：它只留"还没写进长期记忆"的，用途是别丢信息。
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
            del recent[:-_RECENT_KEEP]      # 只留最近这些，别无限涨
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
        """最近 n 轮，**不管进没进长期记忆**。

        跟 ``turns()`` 的区别：那个只留"还没入库"的，一入库就抽走，所以对话历史
        几乎一直是空的——模型看不到刚才说了什么，全靠检索长期记忆去续接。
        短期对话的连贯性（"它"指什么、"刚才那个"是哪个）靠检索是补不回来的。
        """
        with self._lock:
            return list(self._recent.get((session_id, space), []))[-n:]

    def messages(self, session_id: str, space: str, window: int = 0) -> list[dict]:
        """历史渲染成 user/assistant 交替的消息，而不是塞进 system 的一段文本。

        为的是 KV cache：服务端复用的是**最长公共前缀**，所以变的东西必须在后面。
        塞进 system 的话，system 里紧跟着的是每轮都变的记忆，前缀从人设之后就断了，
        历史再长也一个 token 都复用不了。拆成消息之后，前面所有轮次都是稳定前缀，
        每轮只需要算新增的那一条。

        被打断的那轮要标出来：模型看到"你只说到这儿就被打断了"，才不会以为
        自己已经把话说完了。
        """
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
