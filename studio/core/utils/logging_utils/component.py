"""Persistent logging for the web demo without changing existing print calls."""
from __future__ import annotations

import atexit
import os
import platform
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

class _Tee:
    def __init__(self, console, logfile, label: str, lock: threading.RLock, concise=False):
        self.console = console
        self.logfile = logfile
        self.label = label
        self.lock = lock
        self._line_start = True
        self.encoding = getattr(console, "encoding", "utf-8")
        self.errors = getattr(console, "errors", "replace")
        self.concise = concise
        self._console_pending = ""
        self._traceback = False

    def _console_line(self, line):
        clean = _ANSI.sub("", line).strip()
        warning = bool(re.search(r"失败|异常|⚠|\b(?:warning|error|exception|critical|traceback)\b", clean, re.I))
        if clean.startswith("Traceback"):
            self._traceback = True
        if clean.startswith(("[lat]", "[tts-prompt]", "[log]", "[status]")) or warning or self._traceback:
            # Leave the full timing breakdown in the file; hide rolling medians
            # in the terminal. Browser playback latency remains its own line.
            self.console.write(line.split("｜", 1)[0].rstrip() + "\n")
        if self._traceback and clean and not line[:1].isspace() and not clean.startswith("Traceback"):
            self._traceback = False

    def write(self, value) -> int:
        text = str(value)
        with self.lock:
            if not self.concise:
                self.console.write(text)
            else:
                self._console_pending += text.replace("\r", "\n")
                while "\n" in self._console_pending:
                    line, self._console_pending = self._console_pending.split("\n", 1)
                    self._console_line(line)
            clean = _ANSI.sub("", text).replace("\r", "\n")
            for part in clean.splitlines(keepends=True):
                if self._line_start:
                    stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    self.logfile.write(f"{stamp} [{self.label}] ")
                self.logfile.write(part)
                self._line_start = part.endswith(("\n", "\r"))
            self.logfile.flush()
        return len(text)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.console, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.console.fileno()

    def writable(self) -> bool:
        return True

def setup_file_logging(root: Path, requested: str = "", *, concise=False) -> Path:
    """Mirror stdout/stderr to one timestamped UTF-8 file and return its path."""
    if requested:
        path = Path(requested).expanduser()
        if not path.is_absolute():
            path = root / path
    else:
        log_dir = Path(root / 'results' / 'logs')
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = log_dir / f"voicemem-{stamp}-{os.getpid()}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logfile = path.open("a", encoding="utf-8", buffering=1)
    lock = threading.RLock()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(original_stdout, logfile, "stdout", lock, concise=concise)
    sys.stderr = _Tee(original_stderr, logfile, "stderr", lock, concise=concise)

    def close() -> None:
        with lock:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            logfile.flush()
            logfile.close()

    atexit.register(close)

    print(f"[log] 文件：{path}", flush=True)
    print(f"[log] Python={platform.python_version()} pid={os.getpid()} cwd={Path.cwd()}",
          flush=True)
    return path
