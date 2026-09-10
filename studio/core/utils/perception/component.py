from studio.paths import MODELS
"""Studio perception implementation."""
import os
import asyncio
import re
import time
from studio.web import transport as utils

class Perception:
    def _mentioned(self, name: str, text: str) -> bool:
        t = text or ""
        if not name:
            return False
        if name in t:
            return True
        for chunk in re.split(r"[的\s·、,，]+|[^\u4e00-\u9fffA-Za-z0-9]+", name):
            if len(chunk) >= 2 and chunk in t:
                return True
        return False

    def _emotion_by_meaning(self, text: str) -> str:
        import numpy as np
        if not self._PROTO:
            labels, sents = [], []
            for k, vs in self._EMO_PROTO.items():
                labels += [k] * len(vs)
                sents += vs
            self._PROTO["labels"] = labels
            self._PROTO["V"] = np.array(
                utils.shared_embed_model().encode(sents, normalize_embeddings=True),
                dtype=np.float32)
        q = np.array(utils.shared_embed_model().encode([text], normalize_embeddings=True),
                     dtype=np.float32)[0]
        sims = self._PROTO["V"] @ q
        i = int(np.argmax(sims))
        lab, best = self._PROTO["labels"][i], float(sims[i])
        other = [float(sims[j]) for j in range(len(sims)) if self._PROTO["labels"][j] != lab]
        gap = best - (max(other) if other else 0.0)
        return lab if best >= 0.80 and gap >= 0.02 else ""

    def _acoustic_emotion(self, audio_path: str):
        if "m" not in self._E2V:
            from funasr import AutoModel
            self._E2V["m"] = AutoModel(model=str(MODELS / 'emotion2vec'),
                                  hub="hf", disable_update=True)
        r = self._E2V["m"].generate(audio_path, granularity="utterance", extract_embedding=False)
        if not r:
            return "", 0.0
        lab, score = max(zip(r[0]["labels"], r[0]["scores"]), key=lambda x: x[1])
        en = str(lab).split("/")[-1].strip().lower()
        return self._E2V_MAP.get(en, ""), float(score)

    def _sensevoice(self):
        if "t" not in self._SV:
            from voicemem.utils.audio.emotion.detector import shared_transcriber
            self._SV["t"] = shared_transcriber()
        return self._SV["t"]

    def _kick_acoustic(self, send, audio_path: str) -> None:
        if not audio_path:
            return

        async def run():
            try:
                await self.wait_idle("声学情绪")
                t0 = time.monotonic()
                emo, score = await asyncio.to_thread(self._acoustic_emotion, audio_path)
                take = bool(emo) and score >= self.ACOUSTIC_MIN_SCORE and emo in self.ACOUSTIC_TRUST
                if self.BARGE_DEBUG:
                    print(f"  [emotion] 声学(后台) {(time.monotonic()-t0)*1000:.0f}ms "
                          f"-> {emo or '-'} {score:.2f}（{'采纳' if take else '不采纳'}）", flush=True)
                if take:
                    await send({"type": "tag_update", "emotion": emo, "emotion_from": "acoustic"})
            except Exception as e:
                print(f"[web] 后台声学情绪跳过：{type(e).__name__}: {e}", flush=True)

        asyncio.create_task(run())

    def fill_tags(self, payload: dict, text: str, audio_path: str = "",
                  acoustic: bool = True) -> dict:
        """Prepare browser memory labels from retrieval and perception results."""

        for h in payload.get("right_brain_hits") or []:
            claim = h.get("claim") or ""
            if not claim:
                continue
            human = self.rb_human(claim)
            if human and human != claim:

                tail = ""
                for sep in ("｜他说过：", " | he said: "):
                    if sep in h.get("content", ""):
                        tail = sep + h["content"].split(sep, 1)[1]
                        break
                h["content"] = human + tail

        if not payload.get("emotion") and text.strip():
            try:
                from voicemem.rightbrain.anchor_router import normalize_emotion_strict
                payload["emotion"] = normalize_emotion_strict(text) or ""
            except Exception:
                pass

        if not payload.get("emotion") and text.strip():
            try:
                emo = self._emotion_by_meaning(text)
                if emo:
                    payload["emotion"] = emo
                    payload["emotion_from"] = "semantic"
            except Exception as e:
                print(f"[web] 语义情绪跳过：{type(e).__name__}: {e}", flush=True)

        #

        if acoustic and audio_path and True:
            try:
                t0 = time.monotonic()
                emo, score = self._acoustic_emotion(audio_path)
                take = bool(emo) and score >= self.ACOUSTIC_MIN_SCORE and emo in self.ACOUSTIC_TRUST
                if take:
                    payload["emotion"] = emo
                    payload["emotion_from"] = "acoustic"
                if self.BARGE_DEBUG:
                    why = "采纳" if take else ("把握不够" if score < self.ACOUSTIC_MIN_SCORE
                                              else f"{emo} 不在信任名单")
                    print(f"[emotion] 声学 {(time.monotonic()-t0)*1000:.0f}ms "
                          f"-> {emo or '-'} {score:.2f}（{why}）"
                          f"  最终={payload.get('emotion') or '-'}", flush=True)
            except Exception as e:
                print(f"[web] 声学情绪跳过：{type(e).__name__}: {e}", flush=True)

        #

        rb = payload.get("right_brain_hits") or []
        inner = sum(1 for h in rb if h.get("internal"))
        print(f"[hits] 左脑 {len(payload.get('left_brain') or [])} 条  "
              f"右脑 {len(rb)} 条(内部 {inner}，页面显示 {len(rb)-inner})  "
              f"情绪={payload.get('emotion') or '-'}  "
              f"实体={'、'.join(payload.get('entities') or []) or '-'}", flush=True)
        return payload
