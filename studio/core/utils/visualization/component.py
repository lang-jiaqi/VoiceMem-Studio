"""Studio visualization implementation."""
import json
import re
import threading

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

    def memory_changes(self):
        """Return the selected Space and its shared process-local revision feed."""
        from voicemem.utils.common.memory_changes import for_space
        space = self.ACTIVE_SPACE
        directory, _ = self.space_dir(space)
        return space, for_space(directory)

    def owner_name(self, space: str = "", *, memory_vm=None, entries=None, revision=None) -> str:
        """Project the current stored owner name, falling back to voice registration."""
        from voicemem.utils.common.memory_changes import for_space
        from studio.core.utils.visualization.owner_identity import current_owner_name
        from voicemem.utils.common import space as _sp
        space = space or self.ACTIVE_SPACE
        d, _ = self.space_dir(space)
        p = _sp.mm(d, "voiceprint_registry.json")
        if revision is None:
            revision = for_space(d).state()[0]
        try:
            stamp = p.stat().st_mtime_ns
        except FileNotFoundError:
            stamp = None
        cache_key = (revision, stamp)
        cached = self._OWNER_NAME_CACHE.get(space)
        if isinstance(cached, tuple) and cached[0] == cache_key:
            return cached[1]
        name = ""
        try:
            vm = memory_vm or self.vm
            if entries is None:
                entries = vm._o._get_repo()._vector_store.list_entries(user_id=vm._o._user_id)
            name = current_owner_name(entries)
        except Exception as e:
            print(f"[rb] Owner name facts unavailable: {type(e).__name__}: {e}", flush=True)
        try:
            if not name and p.is_file():
                data = json.loads(p.read_text(encoding="utf-8"))
                cands = []
                for v in (data or {}).values():
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
        self._OWNER_NAME_CACHE[space] = (cache_key, name)
        return name

    def _rb_humanize_now(self, claims: list, name: str = "", lang: str = "zh") -> None:
        try:
            from openai import OpenAI
            from voicemem.llm_config import resolve_model
            from studio.harness.speaking_style.policy import RB_HUMANIZE_LANGUAGE_RULES
            who = name or ('the user' if lang == 'en' else '他')
            sysmsg = self._RB_HUMANIZE_PROMPT.format(
                who=who,
                language_rule=RB_HUMANIZE_LANGUAGE_RULES[lang],

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
            with self._RB_HUMAN_LOCK:
                for c, h in zip(claims, lines):
                    self._RB_HUMAN[(lang, name, c)] = h
        except Exception as e:
            print(f"[rb] 判断改写失败：{type(e).__name__}: {e}", flush=True)
        finally:
            with self._RB_HUMAN_LOCK:
                # Resolve failures once so readers share a stable fallback and stop polling.
                for claim in claims:
                    self._RB_HUMAN.setdefault((lang, name, claim), claim)
                self._RB_HUMAN_PENDING.difference_update((lang, name, c) for c in claims)

    def _rb_human_labels(self, claims: list, *, name=None, lang=None, prepare=True) -> dict[str, tuple[str, bool]]:
        """Capture labels and readiness together, sharing pending work across readers."""
        if not self.RB_HUMANIZE:
            return {claim: (claim, False) for claim in claims}
        name = self.owner_name() if name is None else name
        lang = lang or self.SPACE_LANG
        claims = list(dict.fromkeys(claims))
        with self._RB_HUMAN_LOCK:
            todo = [c for c in claims if prepare and c and (lang, name, c) not in self._RB_HUMAN
                    and (lang, name, c) not in self._RB_HUMAN_PENDING]
            self._RB_HUMAN_PENDING.update((lang, name, c) for c in todo)
            labels = {
                c: (self._RB_HUMAN.get((lang, name, c), c),
                    (lang, name, c) in self._RB_HUMAN_PENDING)
                for c in claims
            }
        if todo:
            threading.Thread(target=self._rb_humanize_now, args=(todo, name, lang), daemon=True).start()
        return labels

    def rb_human(self, claim: str) -> str:
        """Return the prepared display label for an internal memory claim."""
        return self._rb_human_labels([claim])[claim][0]

    def rb_human_batch(self, claims: list) -> None:
        """Prepare display labels outside the response hot path."""
        self._rb_human_labels(claims)

    def right_brain_tree(self, uid, facts, *, memory_vm=None, name=None, language=None, include_all=False):
        """Build affective-memory graph data from the active schema."""
        try:
            store = (memory_vm or self.vm)._o._right._traits()
        except Exception as e:
            print(f"[web] 判断表读取失败：{type(e).__name__}: {e}", flush=True)
            return []

        traits = list(store.all(uid, per_slot=10000 if include_all else self.RB_ENTITIES_PER_SLOT))
        labels = self._rb_human_labels([t.claim for t in traits], name=name, lang=language,
                                       prepare=not include_all)
        out = []
        for t in traits:
            out.append({
                "id": t.id,
                "cluster": t.cluster,
                "slot": t.slot,

                "raw": t.claim,
                "text": labels[t.claim][0],
                "label_pending": labels[t.claim][1],
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

    def memory_snapshot(self, limit: int = 48, *, include_all: bool = False, language=None) -> dict:
        """Read a captured Space with request-local labels, preserving stored facts."""
        from voicemem.leftbrain.cognitive_graph.types import SlotV2

        from voicemem.utils.common.memory_changes import for_space
        space = self.ACTIVE_SPACE
        vm = getattr(self, "_SPACES", {}).get(space) or self.vm
        language = language or (self.space_language(space) if hasattr(self, "space_language") else self.SPACE_LANG)
        if language not in {'en', 'zh'}:
            raise ValueError('Display language must be en or zh')
        directory, _ = self.space_dir(space)
        feed = for_space(directory)
        revision, recent = feed.state()
        uid = vm._o._user_id
        left, right = [], []
        total_left, total_right = 0, 0
        entries = []
        hit_ids = set(self._LAST_HIT_IDS)
        try:
            repo = vm._o._get_repo()
            entries = repo._vector_store.list_entries(user_id=uid)

            cog = repo._cognitive_store
            slot_of = {}
            if cog is not None:
                try:
                    for slot in SlotV2:
                        for mid in cog.memory_ids_for_slots(uid, [slot]):
                            slot_of.setdefault(mid, slot.value)
                except Exception as e:
                    print(f"[web] Memory slots unavailable: {type(e).__name__}: {e}", flush=True)

            entries = [e for e in entries if e.get("role") != "assistant"]
            total_left = len(entries)
            rank = {mid: index for index, mid in enumerate(recent)}
            entries.sort(key=lambda e: str(e.get("updated_at") or e.get("date") or ""), reverse=True)
            entries.sort(key=lambda e: (rank.get(str(e["id"]), len(rank)), str(e["id"]) not in hit_ids))
            owner = self.owner_name(space, memory_vm=vm, entries=entries, revision=revision)

            per_slot, kept = {}, []
            for e in entries:
                sl = slot_of.get(e["id"], "daily_life")
                hit = str(e["id"]) in hit_ids
                per_slot[sl] = per_slot.get(sl, 0) + 1
                if include_all or str(e["id"]) in rank or hit or per_slot[sl] <= self.LB_ENTRIES_PER_SLOT:
                    kept.append(e)
            entries = kept

            if include_all:
                visible = entries
            else:
                hits = [e for e in entries if str(e["id"]) in hit_ids]
                rest = [e for e in entries if str(e["id"]) not in hit_ids]
                remaining = max(0, limit - len(hits))
                if rest and str(rest[0]["id"]) in rank:
                    remaining = max(1, remaining)
                visible = hits + rest[:remaining]
            for e in visible:

                d = str(e.get("date", ""))

                try:
                    ents = []
                    for eid in cog.entity_ids_for_memory(e["id"]) or []:
                        ent = cog.get_entity(eid)
                        nm = (getattr(ent, "name", "") if ent else "").strip()
                        ents.append(nm or eid)
                except Exception:
                    ents = []

                left.append({"id": str(e["id"]), "text": e["text"], "date": d if d[:4].isdigit() else "",
                             "slot": slot_of.get(e["id"], "daily_life"),
                             "hit": str(e["id"]) in hit_ids,
                             "entities": list(ents)[:6]})
        except Exception as e:
            print(f"[web] 左脑快照读取失败：{e}", flush=True)
        try:
            owner = self.owner_name(space, memory_vm=vm, entries=entries, revision=revision)
            right = self.right_brain_tree(uid, {}, memory_vm=vm, name=owner,
                                          language=language, include_all=include_all)
            total_right = vm._o._right._traits().counts(uid)[0]
        except Exception as e:
            print(f"[web] 右脑快照读取失败：{e}", flush=True)
        return {"left": left, "right": right, "space": space, "language": language,
                "epoch": feed.epoch, "revision": revision,
                "total": {"left": total_left, "right": total_right}}
