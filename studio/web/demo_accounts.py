"""Small account store and per-account Studio agents for an opt-in public demo."""
from __future__ import annotations

import copy
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from pathlib import Path

COOKIE = "vm_demo_session"
SESSION_SECONDS = 7 * 24 * 60 * 60
_NAME = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fff-]{2,32}$")


class DemoAccounts:
    """Persist credentials and sessions while keeping each user's agent private."""

    def __init__(self, args, *, root=None, agent_factory=None):
        self.root = Path(root or os.environ.get("STUDIO_DEMO_DATA_DIR") or
                         Path.home() / ".local/share/voicemem-studio/demo").expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.db = self.root / "accounts.sqlite"
        self.db.touch(mode=0o600, exist_ok=True)
        self.db.chmod(0o600)
        self.args = args
        self.agent_factory = agent_factory
        self._agents = {}
        self._agent_lock = threading.Lock()
        self._attempts = {}
        self._attempt_lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL UNIQUE,
                    salt BLOB NOT NULL, password_hash BLOB NOT NULL, created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                    expires_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
            """)

    def _connect(self):
        conn = sqlite3.connect(self.db, timeout=10)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)

    def limit_attempt(self, key: str) -> None:
        """Bound account requests from one source without retaining permanent logs."""
        now = time.monotonic()
        with self._attempt_lock:
            if len(self._attempts) > 1000:
                self._attempts = {source: [t for t in times if now - t < 60]
                                  for source, times in self._attempts.items()
                                  if any(now - t < 60 for t in times)}
            recent = [when for when in self._attempts.get(key, ()) if now - when < 60]
            if len(recent) >= 20:
                raise ValueError("尝试过于频繁，请稍后再试。")
            recent.append(now)
            self._attempts[key] = recent

    def register(self, name: str, password: str) -> tuple[str, str]:
        """Create one account with a salted password hash, then issue a session."""
        name = str(name or "").strip()
        if not _NAME.fullmatch(name):
            raise ValueError("名称需为 2–32 个汉字、字母、数字、下划线或连字符。")
        if len(password or "") < 10 or len(password) > 256:
            raise ValueError("密码需为 10–256 个字符。")
        user_id = uuid.uuid4().hex
        salt = secrets.token_bytes(16)
        digest = self._password_hash(password, salt)
        with self._connect() as conn:
            if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] >= 100:
                raise ValueError("体验账号已满。")
            try:
                conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                             (user_id, name, name.casefold(), salt, digest, int(time.time())))
            except sqlite3.IntegrityError:
                raise ValueError("这个名称已被使用。") from None
        return name, self.issue_session(user_id)

    def login(self, name: str, password: str) -> tuple[str, str]:
        """Verify credentials without distinguishing unknown names from bad passwords."""
        with self._connect() as conn:
            row = conn.execute("SELECT id, name, salt, password_hash FROM users WHERE name_key=?",
                               (str(name or "").strip().casefold(),)).fetchone()
        salt = row[2] if row else bytes(16)
        candidate = self._password_hash(str(password or ""), salt)
        if not row or not hmac.compare_digest(candidate, row[3]):
            raise ValueError("名称或密码错误。")
        return row[1], self.issue_session(row[0])

    def issue_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))
            conn.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                         (digest, user_id, int(time.time()) + SESSION_SECONDS))
        return token

    def user(self, token: str) -> tuple[str, str] | None:
        if not token:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute("SELECT users.id, users.name FROM sessions JOIN users "
                               "ON sessions.user_id=users.id WHERE token_hash=? AND expires_at>?",
                               (digest, int(time.time()))).fetchone()
        return tuple(row) if row else None

    def logout(self, token: str) -> None:
        if token:
            with self._connect() as conn:
                conn.execute("DELETE FROM sessions WHERE token_hash=?",
                             (hashlib.sha256(token.encode()).hexdigest(),))

    def agent(self, user_id: str):
        """Reuse one agent per account; all agents share the process model caches."""
        with self._agent_lock:
            if user_id not in self._agents:
                args = copy.copy(self.args)
                args.space = "default"
                args.lang = "zh"
                args.memory_root = ""
                space_root = self.root / "users" / user_id / "spaces"
                space_root.mkdir(mode=0o700, parents=True, exist_ok=True)
                space_root.parent.chmod(0o700)
                space_root.chmod(0o700)
                if self.agent_factory:
                    agent = self.agent_factory(args, space_root)
                else:
                    from studio.core.voiceagent import VoiceAgent
                    agent = VoiceAgent(args, space_root=space_root, public_demo=True)
                    agent.TURN_AUDIO_DIR = self.root / "users" / user_id / "turn_audio"
                self._agents[user_id] = agent
            return self._agents[user_id]
