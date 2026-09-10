"""Studio replay implementation."""
import re
import time
import uuid
from pathlib import Path

class Replay:
    def _musical_memory_ids(self) -> set[str]:
        try:
            tunes = self.vm._o._audio._music_store().list_tunes()
            ids = [f"tune:{t['tune_id']}" for t in tunes]
            if not ids:
                return set()
            store = self.vm._o._get_repo()._cognitive_store
            return set(store.memory_ids_for_slots_v2(self.vm._o._user_id, ids))
        except Exception as e:
            print(f"[web] 读音乐标签失败（不影响回放）：{type(e).__name__}: {e}", flush=True)
            return set()

    def _note_replay(self, memory_id: str) -> None:
        pass
        path = self.audio_of(memory_id)
        if not path:
            return
        try:
            import soundfile as sf
            dur = float(sf.info(path).duration)
        except Exception:
            dur = 15.0
        self._REPLAY_UNTIL = time.monotonic() + dur + 1.0

    def _replaying_now(self) -> bool:
        return time.monotonic() < self._REPLAY_UNTIL

    def _remember_tune(self, audio_path: str) -> None:
        self._LAST_TUNE.update(path=str(audio_path or ""), at=time.monotonic())
        print(f"  [replay] 记住这段音乐：{audio_path}", flush=True)

    def _last_tune_path(self) -> str:
        p = self._LAST_TUNE.get("path") or ""
        if not p or time.monotonic() - float(self._LAST_TUNE.get("at") or 0) > self.LAST_TUNE_TTL_S:
            return ""
        return p if Path(p).exists() else ""

    def _ordinal_of(self, text: str):
        t = text or ""
        for words, n in self._ORDINALS + self._ORDINALS_BACK:
            if any(w in t for w in words):
                return n
        return None

    def _place_of(self, text: str) -> str:
        t = text or ""
        return next((tag for words, tag in self._PLACE_WORDS if any(w in t for w in words)), "")

    def _time_window(self, text: str):
        import re
        from datetime import datetime, timedelta
        t = text or ""
        now = datetime.now()
        day0 = now.replace(hour=0, minute=0, second=0, microsecond=0)
        span = None

        m = re.search(r"(\d+)\s*天前", t)
        if m:
            d = day0 - timedelta(days=int(m.group(1)))
            span = (d, d + timedelta(days=1))

        if span is None:
            wd = next((i for i, words in enumerate(self._WEEKDAYS) if any(w in t for w in words)), None)
            if wd is not None:

                monday = day0 - timedelta(days=day0.weekday())
                if "上上周" in t or "上上星期" in t:
                    monday -= timedelta(days=14)
                elif "上周" in t or "上星期" in t or "上礼拜" in t:
                    monday -= timedelta(days=7)
                d = monday + timedelta(days=wd)
                span = (d, d + timedelta(days=1))

        if span is None:
            m = re.search(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[号日]", t)
            if m:
                mth, dom = int(m.group(1)), int(m.group(2))
                year = now.year - 1 if mth > now.month else now.year
                try:
                    d = datetime(year, mth, dom)
                    span = (d, d + timedelta(days=1))
                except ValueError:
                    span = None
        if span is None:
            m = re.search(r"(?<![0-9])(\d{1,2})\s*[号日](?!\s*[前后])", t)
            if m:
                dom = int(m.group(1))
                try:
                    d = day0.replace(day=dom)
                    if d > day0:
                        d = (day0.replace(day=1) - timedelta(days=1)).replace(day=dom)
                    span = (d, d + timedelta(days=1))
                except ValueError:
                    span = None

        if span is None:
            day = next((d for words, d in self._DAY_WORDS if any(w in t for w in words)), None)
            if day is not None:
                d = day0 + timedelta(days=day)
                span = (d, d + timedelta(days=1))

        if span is None:
            if "上上周" in t or "上上星期" in t:
                monday = day0 - timedelta(days=day0.weekday() + 14)
                span = (monday, monday + timedelta(days=7))
            elif "上周" in t or "上星期" in t or "上礼拜" in t:
                monday = day0 - timedelta(days=day0.weekday() + 7)
                span = (monday, monday + timedelta(days=7))
            elif "这周" in t or "本周" in t or "这星期" in t:
                monday = day0 - timedelta(days=day0.weekday())
                span = (monday, monday + timedelta(days=7))
            elif "上个月" in t or "上月" in t:
                first = day0.replace(day=1)
                span = ((first - timedelta(days=1)).replace(day=1), first)
            elif "这个月" in t or "本月" in t:
                first = day0.replace(day=1)
                span = (first, (first + timedelta(days=32)).replace(day=1))

        hours = next((h for words, h in self._HOUR_WORDS if any(w in t for w in words)), None)
        if span is None and hours is None:
            return None
        if span is None:
            span = (day0, day0 + timedelta(days=1))
        if hours is None:
            return span

        if (span[1] - span[0]).days > 1:
            return span
        return span[0] + timedelta(hours=hours[0]), span[0] + timedelta(hours=hours[1])

    def _tune_memories(self) -> list[dict]:
        from datetime import datetime
        try:
            import sqlite3
            from voicemem.utils.common import space as _space
            c = sqlite3.connect(_space.db(self.vm._o._memory_root))
            c.row_factory = sqlite3.Row
            rows = c.execute(
                """SELECT m.id, m.content, m.created_at,
                          (SELECT group_concat(s.slot) FROM memory_tags s
                            WHERE s.memory_id = m.id AND s.slot LIKE 'scene:%') scenes,
                          (SELECT u.slot FROM memory_tags u
                            WHERE u.memory_id = m.id AND u.slot LIKE 'tune:%' LIMIT 1) tune,
                          EXISTS (SELECT 1 FROM memory_tags o
                                   WHERE o.memory_id = m.id AND o.slot = 'sound_only') sound_only
                     FROM memories m
                     WHERE EXISTS (SELECT 1 FROM memory_tags t
                                    WHERE t.memory_id = m.id
                                      AND (t.slot LIKE 'tune:%' OR t.slot = 'sound_only'))
                     ORDER BY m.created_at DESC LIMIT 300""").fetchall()
            c.close()
        except Exception as e:
            print(f"[replay] 列音乐记忆失败：{type(e).__name__}: {e}", flush=True)
            return []

        out = []
        for r in rows:
            if not self.audio_of(r["id"]):
                continue
            try:
                at = datetime.fromisoformat(r["created_at"]).astimezone().replace(tzinfo=None)
            except Exception:
                continue
            out.append({"id": r["id"], "at": at, "text": r["content"] or "",
                        "sound_only": bool(r["sound_only"]),
                        "tune": (r["tune"] or "").split(":", 1)[-1],
                        "scenes": {x.split(":", 1)[1] for x in (r["scenes"] or "").split(",") if ":" in x}})
        return out

    def _archived_memory_ids(self) -> list[str]:
        try:
            import sqlite3
            from voicemem.utils.common import space as _space
            c = sqlite3.connect(_space.db(self.vm._o._memory_root))
            rows = c.execute("SELECT id FROM memories ORDER BY created_at DESC LIMIT 200").fetchall()
            c.close()
            return [r[0] for r in rows]
        except Exception as e:
            print(f"[web] 列存档记忆失败（不影响回放）：{type(e).__name__}: {e}", flush=True)
            return []

    def _created_at(self, mid: str):
        from datetime import datetime
        try:
            import sqlite3
            from voicemem.utils.common import space as _space
            c = sqlite3.connect(_space.db(self.vm._o._memory_root))
            row = c.execute("SELECT created_at FROM memories WHERE id=?", (mid,)).fetchone()
            c.close()
            if not row or not row[0]:
                return None
            return datetime.fromisoformat(row[0]).astimezone().replace(tzinfo=None)
        except Exception:
            return None

    def _stitch(self, memory_ids: list) -> str:
        paths = [q for q in (self.audio_of(m) for m in memory_ids) if q]
        if not paths:
            return ""
        if len(paths) == 1:
            return paths[0]
        key = "|".join(paths)
        if key in self._STITCH_CACHE and Path(self._STITCH_CACHE[key]).exists():
            return self._STITCH_CACHE[key]
        try:
            import numpy as np
            import soundfile as sf
            from voicemem.utils.audio.stream_io import resample as _resample
            chunks, sr0 = [], None
            for q in paths:
                x, sr = sf.read(q, dtype="float32", always_2d=False)
                if getattr(x, "ndim", 1) > 1:
                    x = x.mean(axis=1)
                if sr0 is None:
                    sr0 = sr
                elif sr != sr0:
                    x = _resample(x, sr, sr0)
                chunks.append(x)
            out = self.TURN_AUDIO_DIR / f"stitch_{uuid.uuid4().hex[:12]}.wav"
            sf.write(out, np.concatenate(chunks), sr0)
            self._STITCH_CACHE[key] = str(out)
            total = sum(len(c) for c in chunks) / float(sr0 or 16000)
            print(f"  [replay] 拼好 {len(paths)} 段 → {total:.1f}s", flush=True)
            return str(out)
        except Exception as e:
            print(f"  [replay] 拼接失败，只放第一段：{type(e).__name__}: {e}", flush=True)
            return paths[0]

    def _same_song_group(self, pool: list, pick: dict) -> list:
        tune = pick.get("tune") or ""
        same = [x for x in pool
                if (x.get("tune") or "") == tune and (tune != "unidentified" or x.get("sound_only"))]
        same.sort(key=lambda x: x["at"])
        if pick not in same:
            return [pick]
        i = same.index(pick)
        lo = i
        while lo > 0 and (same[lo]["at"] - same[lo - 1]["at"]).total_seconds() <= self.TUNE_GAP_S:
            lo -= 1
        hi = i
        while hi + 1 < len(same) and (same[hi + 1]["at"] - same[hi]["at"]).total_seconds() <= self.TUNE_GAP_S:
            hi += 1
        return same[lo:hi + 1]

    def _group_id(self, group: list) -> str:
        if not group:
            return ""
        if len(group) == 1:
            return group[0]["id"]
        return self.GROUP_ID_PREFIX + ",".join(x["id"] for x in group)

    def _prefer_sound_only(self, cand: list) -> list:
        only = [x for x in cand if x.get("sound_only")]
        return only or cand

    def _replay_id(self, text: str, result) -> str:
        if not self._wants_sound(text):
            return ""

        pool = self._tune_memories()
        window, place = self._time_window(text), self._place_of(text)
        nth = self._ordinal_of(text)

        if nth is not None and window is None:
            from datetime import datetime, timedelta
            day0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            window = (day0, day0 + timedelta(days=1))

        if pool and (window or place):
            cand = pool
            if window:
                cand = [x for x in cand if window[0] <= x["at"] <= window[1]]
            if place:
                cand = [x for x in cand if place in x["scenes"]]
            cand = self._prefer_sound_only(cand)
            if cand:
                cand.sort(key=lambda x: x["at"])
                if nth is None:
                    pick = cand[-1]
                elif -len(cand) <= (nth - 1 if nth > 0 else nth) < len(cand):
                    pick = cand[nth - 1 if nth > 0 else nth]
                else:
                    print(f"  [replay] 那段时间只有 {len(cand)} 首，没有第 {nth} 首", flush=True)
                    return ""
                print(f"  [replay] 按条件挑中 {pick['id'][:12]}（{pick['at']:%m-%d %H:%M}"
                      f"{' / ' + place if place else ''}"
                      f"{' / 第 %d 首' % nth if nth else ''}，共 {len(cand)} 首）", flush=True)
                return self._group_id(self._same_song_group(pool, pick))
            print(f"  [replay] 池里 {len(pool)} 段音乐，没有符合条件的"
                  f"（{'时间 %s~%s ' % (window[0], window[1]) if window else ''}"
                  f"{'地点 ' + place if place else ''}）", flush=True)
            return ""

        if pool:
            pick = max(self._prefer_sound_only(pool), key=lambda x: x["at"])
            print(f"  [replay] 没说条件，放最近的 {pick['id'][:12]}"
                  f"（{pick['at']:%m-%d %H:%M}）", flush=True)
            return self._group_id(self._same_song_group(pool, pick))

        playable = [h.memory_id for h in (getattr(result, "hits", None) or [])
                    if self.audio_of(h.memory_id)]
        if playable:
            print(f"  [replay] 没有音乐标签，放检索到的 {playable[0][:12]}", flush=True)
            return playable[0]
        if self._last_tune_path():
            print("  [replay] 还没入库，放刚听过的那段", flush=True)
            return self.LAST_TUNE_ID
        print(f"  [replay] 放不了：一段音乐记忆都没有，检索 "
              f"{len(getattr(result, 'hits', None) or [])} 条也都没音频，"
              f"缓存 path={self._LAST_TUNE.get('path') or '(空)'}", flush=True)
        return ""

    def _wants_sound(self, text: str) -> bool:
        return any(w in (text or "") for w in self._SOUND_WORDS)
