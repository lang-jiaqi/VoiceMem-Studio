"""Studio spaces implementation."""
import time
from pathlib import Path
from studio.core.utils.dialogue.component import CONTEXT
from studio.core.voicemem import open_memory
from studio.paths import ROOT as _ROOT

class Spaces:
    def space_language(self, name: str) -> str:
        """Read the Space's initial language fallback, defaulting to English."""
        import json as _json
        d, safe = self.space_dir(name)
        f = d / f"{safe}.json"
        try:
            v = (_json.loads(f.read_text(encoding="utf-8"))
                 .get("space", {}).get("language", ""))
            return "zh" if str(v).lower().startswith("zh") else "en"
        except Exception:
            return "en"

    def _write_space_language(self, name: str, lang: str) -> None:
        import json as _json
        d, safe = self.space_dir(name)
        f = d / f"{safe}.json"
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
        root = getattr(self, "_SPACE_ROOT", None) or _ROOT / "voicemem_memoryspace"
        return root / safe, safe

    def get_space(self, name: str):
        """Open one memory instance per space, preserving its stored language."""
        directory, safe = self.space_dir(name)
        if safe not in self._SPACES:
            if not directory.exists() or not any(directory.iterdir()):
                directory.mkdir(mode=0o700 if getattr(self, "_SPACE_ROOT", None) else 0o777,
                                parents=True, exist_ok=True)
                self._write_space_language(safe, self.ARGS.lang)
            cfg = dict(self.CONFIG)
            cfg["space"] = safe
            if getattr(self, "_SPACE_ROOT", None) is not None:
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
            if self.ARGS.llm == "local":
                self._LOCAL_LLM = inst._reply_src
            self._SPACES[safe] = inst
            print(f"[space] 打开「{safe}」用了 {time.monotonic()-t0:.1f}s", flush=True)
        return self._SPACES[safe]

    def use_space(self, name: str) -> str:
        """Select the active memory instance and its ambiguous-input fallback."""
        self.vm = self.get_space(name)
        _, self.ACTIVE_SPACE = self.space_dir(name)
        self.SPACE_LANG = self.space_language(self.ACTIVE_SPACE)
        if self._LOCAL_LLM is not None:
            self._LOCAL_LLM.system = self._rt_persona(self.SPACE_LANG)
        return self.ACTIVE_SPACE

    def list_spaces(self) -> list:
        """List available spaces and read their memory counts without model work."""
        import sqlite3
        root = getattr(self, "_SPACE_ROOT", None) or _ROOT / "voicemem_memoryspace"
        out = []
        for d in sorted(p for p in root.glob("*") if p.is_dir()):
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
            out.append({"id": d.name, "name": d.name, "count": n,
                        "active": d.name == self.ACTIVE_SPACE, "open": d.name in self._SPACES,
                        "language": self.space_language(d.name)})
        return out

    def create_space(self, name: str, language: str = "") -> dict:
        """Create and warm an empty space; reject an existing nonempty space."""
        d, safe = self.space_dir(name)
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f"「{safe}」已经存在了")
        d.mkdir(parents=True, exist_ok=True)
        lang = "zh" if getattr(self, "PUBLIC_DEMO", False) or str(language or self.ARGS.lang).lower().startswith("zh") else "en"

        self._write_space_language(safe, lang)
        self.get_space(safe)
        print(f"[space] 新建「{safe}」（语言 {lang}）→ {d}", flush=True)
        return {"id": safe, "name": safe, "count": 0, "language": lang}
