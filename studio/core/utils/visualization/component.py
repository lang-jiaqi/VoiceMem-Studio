"""Studio visualization implementation."""
import json
import re
import threading
from studio.web import transport as utils

class Visualization:
    def rb_cluster(self, content: str, memory_class: str = "", emotion: str = "") -> str:
        """Map affective memory metadata into a browser cluster."""

        text = re.sub(r"^\s*\[[0-9-]{6,12}\]\s*", "", content or "")
        head = text[:14]
        for slot, cluster in self.SLOT_TO_CLUSTER.items():
            if slot in head:
                return cluster
        if str(memory_class) == "response_experience":
            return "experiences"
        if emotion not in self._CALM:
            return "emotion"
        return "experiences"

    def audio_of(self, memory_id: str) -> str:
        """Resolve an archived recording for a memory ID or return an empty path."""
        if memory_id == self.LAST_TUNE_ID:
            return self._last_tune_path()
        if memory_id.startswith(self.GROUP_ID_PREFIX):
            return self._stitch([m for m in memory_id[len(self.GROUP_ID_PREFIX):].split(",") if m])
        try:
            r = self.vm._o._audio.GetOriginalAudio(memory_id)
            return r.get("audio_path") or "" if r.get("found") else ""
        except Exception as e:
            print(f"[web] 查存档音频失败：{e}", flush=True)
            return ""

    def hit_cluster(self, content: str, source: str) -> str:
        """Map a retrieval hit into a browser cluster."""
        return self.rb_cluster(content, source, "")

    def _rb_cluster(self, m) -> str:
        meta = getattr(m, "metadata", None) or {}
        return self.rb_cluster(getattr(m, "content", ""),
                          str(getattr(m, "memory_class", "")),
                          meta.get("emotion", ""))

    def fact_index(self, uid: str) -> dict:
        """Return a memory-ID-to-text index for the requested user."""
        try:
            entries = self.vm._o._get_repo()._vector_store.list_entries(user_id=uid)
            return {e["id"]: e["text"] for e in entries}
        except Exception as e:
            print(f"[web] 读左脑事实失败：{e}", flush=True)
            return {}

    def owner_name(self, space: str = "") -> str:
        """Read the owner name from the selected space without guessing identity."""
        space = space or self.ACTIVE_SPACE
        if space in self._OWNER_NAME_CACHE:
            return self._OWNER_NAME_CACHE[space]
        name = ""
        try:
            import json
            from voicemem.utils.common import space as _sp
            d, _ = self.space_dir(space)
            p = _sp.mm(d, "voiceprint_registry.json")
            if p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                cands = []
                for key, v in (data or {}).items():
                    if not isinstance(v, dict) or v.get("role") != "user":
                        continue
                    n = (v.get("name") or "").strip()

                    if not n or n.lower() == "user" or any(c in n for c in self._BAD_NAME_CHARS):
                        continue
                    cands.append((bool(v.get("entity_id")), n))
                if cands:

                    cands.sort(key=lambda t: not t[0])
                    name = cands[0][1]
        except Exception as e:
            print(f"[rb] 主人姓名读取失败：{type(e).__name__}: {e}", flush=True)
        self._OWNER_NAME_CACHE[space] = name
        return name

    def _rb_humanize_now(self, claims: list, name: str = "", lang: str = "zh") -> None:
        try:
            from openai import OpenAI
            from voicemem.llm_config import resolve_model
            who = name or "他"
            sysmsg = self._RB_HUMANIZE_PROMPT.format(
                who=who,

                name_rule=(f"· **每条都直接叫他「{name}」**，别用「他」代替——"
                           "页面上每条是独立一行，点名是在说「这是关于谁的」。\n"
                           if name else ""),
                examples=self._RB_HUMANIZE_EXAMPLES
                         .replace("{who}", who))
            r = OpenAI().chat.completions.create(
                model=resolve_model(role="chat"), temperature=0.7,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": "\n".join(claims)}],
            )
            lines = [x.strip() for x in (r.choices[0].message.content or "").splitlines() if x.strip()]
            if len(lines) != len(claims):
                print(f"[rb] 改写行数不符（{len(lines)}≠{len(claims)}），这批跳过", flush=True)
                return
            for c, h in zip(claims, lines):
                self._RB_HUMAN[(lang, name, c)] = h
        except Exception as e:
            print(f"[rb] 判断改写失败：{type(e).__name__}: {e}", flush=True)
        finally:
            self._RB_HUMAN_PENDING.difference_update((lang, name, c) for c in claims)

    def rb_human(self, claim: str) -> str:
        """Return the prepared display label for an internal memory claim."""
        if not self.RB_HUMANIZE or not claim:
            return claim
        name, lang = self.owner_name(), self.SPACE_LANG
        hit = self._RB_HUMAN.get((lang, name, claim))
        if hit:
            return hit
        if (lang, name, claim) not in self._RB_HUMAN_PENDING:
            self._RB_HUMAN_PENDING.add((lang, name, claim))
            import threading
            threading.Thread(target=self._rb_humanize_now,
                             args=([claim], name, lang), daemon=True).start()
        return claim

    def rb_human_batch(self, claims: list) -> None:
        """Prepare display labels outside the response hot path."""
        if not self.RB_HUMANIZE:
            return
        name, lang = self.owner_name(), self.SPACE_LANG
        todo = [c for c in dict.fromkeys(claims)
                if c and (lang, name, c) not in self._RB_HUMAN
                and (lang, name, c) not in self._RB_HUMAN_PENDING]
        if not todo:
            return
        self._RB_HUMAN_PENDING.update((lang, name, c) for c in todo)
        import threading
        threading.Thread(target=self._rb_humanize_now, args=(todo, name, lang), daemon=True).start()

    def right_brain_tree(self, uid, facts):
        """Build affective-memory graph data from the active schema."""
        try:
            store = self.vm._o._right._traits()
        except Exception as e:
            print(f"[web] 判断表读取失败：{type(e).__name__}: {e}", flush=True)
            return []

        traits = list(store.all(uid, per_slot=self.RB_ENTITIES_PER_SLOT))
        self.rb_human_batch([t.claim for t in traits])
        out = []
        for t in traits:
            out.append({
                "cluster": t.cluster,
                "slot": t.slot,

                "raw": t.claim,
                "text": self.rb_human(t.claim),
                "desc": "",
                "notes": [{"text": e.quote, "emotion": e.emotion, "cause": e.cause}
                          for e in t.evidence],
            })
        return out

    def _right_brain_tree_v1(self, uid: str, facts: dict) -> list:
        graph = self.vm._o._right._rb_graph_store()
        repo = self.vm._o._right._rb_repo()
        notes = {}
        for m in repo.list_all(uid):

            if getattr(m, "memory_class", "") == "response_experience":
                continue
            meta = getattr(m, "metadata", None) or {}
            notes[m.id] = {
                "text": m.content,
                "emotion": meta.get("emotion", ""),
                "cause": facts.get(meta.get("left_memory_id", ""), ""),
            }

        out = []
        for slot in graph.list_slots(uid):

            cluster = self.SLOT_TO_CLUSTER.get(slot.name, "experiences")

            #

            def rows(newest):
                out = []
                for ent in graph.get_entities_for_slot(uid, slot.id, newest_first=newest):
                    mids = [i for i in graph.get_memories_for_entity(ent.id) if i in notes]
                    out.append((len(mids), ent, mids))
                return out

            fresh_n = max(1, self.RB_ENTITIES_PER_SLOT // 2)
            by_recent = rows(True)[:fresh_n]
            taken = {t[1].id for t in by_recent}
            by_evidence = [t for t in sorted(rows(False), key=lambda t: -t[0])
                           if t[1].id not in taken]
            picked = by_recent + by_evidence[:self.RB_ENTITIES_PER_SLOT - len(by_recent)]

            for _, ent, mids in picked:

                if not mids:
                    continue
                ns = [notes[i] for i in mids]

                #

                desc = (getattr(ent, "description", "") or "").strip()
                out.append({
                    "cluster": cluster,
                    "slot": slot.name,
                    "text": ent.name,
                    "desc": desc,
                    "notes": ns,
                })
        return out

    def note_hits(self, result) -> None:
        """Record retrieved memory IDs so visible hits survive display limits."""
        self._LAST_HIT_IDS.clear()
        for h in (getattr(result, "hits", None) or []):
            mid = getattr(h, "memory_id", "")
            if mid:
                self._LAST_HIT_IDS.add(str(mid))

    def memory_snapshot(self, limit: int = 48) -> dict:
        """Read the current memory graph without running inference."""
        from voicemem.leftbrain.cognitive_graph.types import SlotV2

        uid = self.vm._o._user_id
        left, right = [], []
        try:
            repo = self.vm._o._get_repo()
            entries = repo._vector_store.list_entries(user_id=uid)

            cog = repo._cognitive_store
            slot_of = {}
            for slot in SlotV2:
                for mid in cog.memory_ids_for_slots(uid, [slot]):
                    slot_of.setdefault(mid, slot.value)

            entries = [e for e in entries if e.get("role") != "assistant"]

            #

            per_slot, kept = {}, []
            hit_first = ([e for e in entries if str(e["id"]) in self._LAST_HIT_IDS] +
                         [e for e in entries if str(e["id"]) not in self._LAST_HIT_IDS])
            for e in hit_first:
                sl = slot_of.get(e["id"], "daily_life")
                hit = str(e["id"]) in self._LAST_HIT_IDS
                per_slot[sl] = per_slot.get(sl, 0) + 1
                if hit or per_slot[sl] <= self.LB_ENTRIES_PER_SLOT:
                    kept.append(e)
            entries = kept

            head = [e for e in entries if str(e["id"]) in self._LAST_HIT_IDS]
            rest = [e for e in entries if str(e["id"]) not in self._LAST_HIT_IDS]
            for e in (head + rest)[:max(limit, len(head))]:

                d = str(e.get("date", ""))

                try:
                    ents = []
                    for eid in cog.entity_ids_for_memory(e["id"]) or []:
                        ent = cog.get_entity(eid)
                        nm = (getattr(ent, "name", "") if ent else "").strip()
                        ents.append(nm or eid)
                except Exception:
                    ents = []

                left.append({"text": e["text"], "date": d if d[:4].isdigit() else "",
                             "slot": slot_of.get(e["id"], "daily_life"),
                             "hit": str(e["id"]) in self._LAST_HIT_IDS,
                             "entities": list(ents)[:6]})
        except Exception as e:
            print(f"[web] 左脑快照读取失败：{e}", flush=True)
        try:
            right = self.right_brain_tree(uid, self.fact_index(uid))
        except Exception as e:
            print(f"[web] 右脑快照读取失败：{e}", flush=True)
        return {"left": left, "right": right}
