"""Studio spaces implementation."""
import time
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from studio.core.utils.dialogue.component import CONTEXT
from studio.core.voicemem import open_memory
from studio.paths import ROOT as _ROOT


@dataclass(frozen=True)
class SpaceSelection:
    """An atomically published selection; preparing another Space leaves it intact."""

    name: str = ""
    memory: object = None
    language: str = "en"
    local_reply: object = None


_EMPTY_SELECTION = SpaceSelection()


def capture_space(owner):
    """Capture matching Space and memory ownership, including injected agents."""
    selected = getattr(owner, "_SPACE_SELECTION", None)
    if isinstance(selected, SpaceSelection):
        return selected.name, selected.memory
    return owner.ACTIVE_SPACE, owner.vm


class Spaces:
    """Own Space preparation and publish selection without blocking readers."""

    def _space_lock(self):
        return self.__dict__.setdefault("_SPACES_LOCK", threading.RLock())

    @property
    def ACTIVE_SPACE(self):
        return getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION).name

    @ACTIVE_SPACE.setter
    def ACTIVE_SPACE(self, value):
        self._SPACE_SELECTION = replace(getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION), name=value)

    @property
    def vm(self):
        return getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION).memory

    @vm.setter
    def vm(self, value):
        self._SPACE_SELECTION = replace(getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION), memory=value)

    @property
    def SPACE_LANG(self):
        return getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION).language

    @SPACE_LANG.setter
    def SPACE_LANG(self, value):
        self._SPACE_SELECTION = replace(getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION), language=value)

    @property
    def _LOCAL_LLM(self):
        return getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION).local_reply

    @_LOCAL_LLM.setter
    def _LOCAL_LLM(self, value):
        self._SPACE_SELECTION = replace(getattr(self, "_SPACE_SELECTION", _EMPTY_SELECTION), local_reply=value)

    def _fixed_memory_root(self):
        """Keep an explicit memory_root as one database, never a Space parent."""
        configured = getattr(self, "CONFIG", {}).get("memory_root")
        return Path(configured).expanduser().resolve() if configured else None

    def space_language(self, name: str) -> str:
        """Read the Space's initial language fallback, defaulting to English."""
        import json as _json
        d, safe = self.space_dir(name)
        from voicemem.utils.common import space as _sp
        f = _sp.json_path(d)
        try:
            v = (_json.loads(f.read_text(encoding="utf-8"))
                 .get("space", {}).get("language", ""))
            return "zh" if str(v).lower().startswith("zh") else "en"
        except Exception:
            return "en"

    def _write_space_language(self, name: str, lang: str) -> None:
        import json as _json
        d, safe = self.space_dir(name)
        from voicemem.utils.common import space as _sp
        f = _sp.json_path(d)
        try:
            doc = _json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
            doc.setdefault("space", {})["language"] = lang
            f.write_text(_json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[space] 写语言失败（不影响使用）：{e}", flush=True)

    def _space_is_empty(self, name: str) -> bool:
        try:
            import sqlite3
            from voicemem.utils.common import space as _sp
            d, _ = self.space_dir(name)
            db = _sp.db(d)
            if not Path(db).exists():
                return True
            c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            except sqlite3.OperationalError:

                return True
            finally:
                c.close()
            return n == 0
        except Exception:
            return False

    def set_lang(self, lang: str) -> str:
        """Change UI language without reopening memory or changing conversation language."""
        self.UI_LANG = "en" if str(lang).lower().startswith("en") else "zh"
        return self.UI_LANG

    def space_dir(self, name: str):
        """Return a sanitized space directory and name; reject an empty name."""
        import re as _re
        safe = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "", (name or "").strip())[:32]
        if not safe:
            raise ValueError("空间名字不能为空")
        fixed = self._fixed_memory_root()
        if fixed is not None:
            configured = _re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]", "", self.ARGS.space.strip())[:32]
            if safe != configured:
                raise ValueError("--memory_root 指定的是单个记忆库，不能切换到其他空间")
            return fixed, safe
        root = _ROOT / "voicemem_memoryspace"
        return root / safe, safe

    def get_space(self, name: str):
        """Open one memory instance per space, preserving its stored language."""
        with self._space_lock():
            return self._open_space(name)

    def _open_space(self, name: str):
        directory, safe = self.space_dir(name)
        if safe not in self._SPACES:
            if not directory.exists() or not any(directory.iterdir()):
                directory.mkdir(parents=True, exist_ok=True)
                self._write_space_language(safe, self.ARGS.lang)
            cfg = dict(self.CONFIG)
            cfg["space"] = safe
            if self._fixed_memory_root() is not None:
                cfg["memory_root"] = str(directory)
            cfg["memory_language"] = self.space_language(safe)

            lang = self.space_language(safe)

            if isinstance(self.CONFIG["reply"], dict):
                cfg["reply"] = {**self.CONFIG["reply"],
                                "llm": {**self.CONFIG["reply"]["llm"],
                                        "config": {**self.CONFIG["reply"]["llm"]["config"],
                                                   "system": self._rt_persona(lang)}}}
            elif hasattr(self.CONFIG["reply"], "set_system"):
                self.CONFIG["reply"].set_system(self._rt_persona(lang))
            t0 = time.monotonic()
            inst = open_memory(cfg)
            self._SPACES[safe] = inst
            print(f"[space] 打开「{safe}」用了 {time.monotonic()-t0:.1f}s", flush=True)
        return self._SPACES[safe]

    def use_space(self, name: str) -> str:
        """Select the active memory instance and its ambiguous-input fallback."""
        with self._space_lock():
            memory = self.get_space(name)
            _, safe = self.space_dir(name)
            language = self.space_language(safe)
            local_reply = memory._reply_src if self.ARGS.llm == "local" else None
            if local_reply is not None:
                local_reply.system = self._rt_persona(language)
            self._SPACE_SELECTION = SpaceSelection(safe, memory, language, local_reply)
            return safe

    def list_spaces(self) -> list:
        """List available spaces and read their memory counts without model work."""
        import sqlite3
        fixed = self._fixed_memory_root()
        if fixed is not None:
            directory, safe = self.space_dir(self.ARGS.space)
            directories = [(directory, safe)]
        else:
            root = _ROOT / "voicemem_memoryspace"
            directories = [(d, d.name) for d in sorted(p for p in root.glob("*") if p.is_dir())]
        active = self.ACTIVE_SPACE
        out = []
        for d, name in directories:
            n = 0
            try:
                from voicemem.utils.common import space as _sp
                db = _sp.db(d)
                if Path(db).exists():
                    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
                    n = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                    c.close()
            except Exception:
                n = 0
            out.append({"id": name, "name": name, "count": n,
                        "active": name == active, "open": name in self._SPACES,
                        "language": self.space_language(name), "fixed": fixed is not None})
        return out

    def create_space(self, name: str, language: str = "") -> dict:
        """Create and warm an empty space; reject an existing nonempty space."""
        with self._space_lock():
            return self._create_space(name, language)

    def _create_space(self, name: str, language: str) -> dict:
        if self._fixed_memory_root() is not None:
            raise ValueError("--memory_root 指定的是单个记忆库，不能新建空间")
        d, safe = self.space_dir(name)
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"「{safe}」已经存在了")
        d.mkdir(parents=True, exist_ok=True)
        lang = "zh" if str(language or self.ARGS.lang).lower().startswith("zh") else "en"

        self._write_space_language(safe, lang)
        self.get_space(safe)
        print(f"[space] 新建「{safe}」（语言 {lang}）→ {d}", flush=True)
        return {"id": safe, "name": safe, "count": 0, "language": lang}
