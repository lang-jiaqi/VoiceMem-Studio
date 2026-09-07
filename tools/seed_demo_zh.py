"""Prepare/verify ten synthetic memories, then replace ONLY a stopped demo-zh.

prepare creates a separate staging space. install atomically renames the old
space into results/backups and replaces it with the verified staging space.
No original files are deleted. The server must be stopped before install.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TARGET = ROOT / "voicemem_memoryspace/demo-zh"
FIXTURE = ROOT / "evals/fixtures/demo_zh_clean10.json"
STAGING = ROOT / "results/seed-staging"


def prepare():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["MEM0_TELEMETRY"] = "False"
    # mem0 constructs an unused OpenAI client even for infer=False. Use an
    # explicitly invalid local-only placeholder, never a real credential.
    os.environ["OPENAI_API_KEY"] = "offline-seed-no-api"
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:1"
    os.environ["VOICEMEM_MODELS_DIR"] = str(ROOT / "models")
    import torch
    torch.set_num_threads(2)
    from sentence_transformers import SentenceTransformer
    from voicemem.leftbrain.local_embedder import LocalEmbedder
    from voicemem.leftbrain.memory_repository import LeftBrainMemoryRepository, LeftBrainMemoryRepositoryConfig
    from voicemem.leftbrain.memory_repository_v2 import LeftBrainMemoryRepositoryV2
    from voicemem.leftbrain.extract_facts_openai import ExtractedAdditiveMemory
    from voicemem.leftbrain.cognitive_graph.types import AnnotatedFact, EntityAnnotation, EntityType, SlotV2
    from voicemem.utils.common.space import describe

    rows = json.loads(FIXTURE.read_text())["memories"]
    assert len(rows) == 10 and len({r["text"] for r in rows}) == 10
    STAGING.mkdir(parents=True, exist_ok=True)
    parent = Path(tempfile.mkdtemp(prefix="clean10-", dir=STAGING))
    stage = parent / "demo-zh"
    stage.mkdir()
    (stage / "multi_modal").mkdir()
    meta = json.loads((TARGET / "demo-zh.json").read_text())
    assert meta["mem0"]["model"] == "e5" and meta["mem0"]["dims"] == 384
    uid = meta["space"]["user_id"]
    # Preserve the language/user/model contract, not old runtime metadata.
    describe(stage, user_id=uid, mode="multi_modal", dims=384, embed_model="e5")
    descriptor = stage / "demo-zh.json"
    data = json.loads(descriptor.read_text())
    data["space"]["language"] = "zh"
    descriptor.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    model = SentenceTransformer(str(ROOT / "models/embedding"), device="cpu", local_files_only=True)
    class CpuEmbedder(LocalEmbedder):
        def _encode(self, texts, prefix):
            return model.encode([prefix + t for t in texts], normalize_embeddings=True)
        def embed_query_text(self, text):
            return self._encode([text], self.model.query_prefix)[0].tolist()
    embedder = CpuEmbedder("e5", language="zh")
    class FixedAnnotator:
        def annotate(self, texts):
            by_text = {r["text"]: r for r in rows}
            return [AnnotatedFact(text, SlotV2(by_text[text]["slot"]), [
                EntityAnnotation(by_text[text]["entity"], EntityType(by_text[text]["entity_type"]))])
                for text in texts]
    repo = LeftBrainMemoryRepositoryV2(embedder, config=LeftBrainMemoryRepositoryConfig(
        json_path=descriptor, db_path=stage / "demo-zh.sqlite",
        cognitive_db_path=stage / "demo-zh.sqlite", enable_cognitive_graph=True),
        cognitive_annotator=FixedAnnotator())
    # Parent append does exact infer=False storage and deterministic graph
    # annotations; manually assign fixed tags instead of another model pass.
    ids = LeftBrainMemoryRepository.append_extracted(repo, [
        ExtractedAdditiveMemory(r["id"], r["text"], "user") for r in rows],
        user_id=uid, extra_metadata={"synthetic": True, "fixture": "demo_zh_clean10"})
    assert len(ids) == 10
    for mid, row in zip(ids, rows):
        repo._cognitive_store.upsert_memory_tags(mid, uid, [(row["slot"], 1.0)])
    entries = repo.vector_store.list_entries(user_id=uid)
    assert len(entries) == 10
    assert {e["id"] for e in entries} == set(ids)
    checks = []
    for mid, row in zip(ids, rows):
        t0 = time.perf_counter()
        hits = repo.search(row["question"], user_id=uid, top_k=3)
        hit_ids = [h.memory_id for h in hits]
        rank = hit_ids.index(mid) + 1 if mid in hit_ids else None
        checks.append({"fixture_id": row["id"], "memory_id": mid,
                       "rank_top3": rank, "search_ms": round((time.perf_counter()-t0)*1000, 1)})
        print(f"{row['id']} rank_top3={rank} query={row['question']}", flush=True)
    assert all(c["rank_top3"] is not None for c in checks), "Seed recall check failed; original untouched"
    report = {"stage": str(stage), "count": 10, "user_id": uid,
              "fixture": str(FIXTURE), "checks": checks}
    (parent / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    repo.vector_store._mem0.vector_store.client.close()
    with sqlite3.connect(stage / "demo-zh.sqlite") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT count(*) FROM memories").fetchone()[0] == 10
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    print(f"PREPARED={stage}", flush=True)


def install(stage):
    stage = Path(stage).resolve()
    assert stage.parent.parent == STAGING.resolve() and stage.name == "demo-zh"
    assert TARGET.resolve() == ROOT / "voicemem_memoryspace/demo-zh" and not TARGET.is_symlink()
    report = json.loads((stage.parent / "verification.json").read_text())
    assert report["stage"] == str(stage) and report["count"] == 10
    verify(stage)
    # Refuse replacement while any process has this space open, even if the
    # server is using a different port or has background workers after Ctrl+C.
    opened = subprocess.run(["lsof", "-t", "+D", str(TARGET)], capture_output=True, text=True)
    if opened.returncode not in (0, 1) or opened.stdout.strip():
        raise SystemExit("demo-zh still open by processes: " + opened.stdout.strip())
    if opened.stderr.strip():
        raise SystemExit("Cannot verify closed space: " + opened.stderr.strip())
    if (TARGET / ".gitkeep").is_file():
        shutil.copy2(TARGET / ".gitkeep", stage / ".gitkeep")
    backup = ROOT / "results/backups" / ("demo-zh-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    backup.parent.mkdir(parents=True, exist_ok=True)
    assert not backup.exists()
    TARGET.rename(backup)
    try:
        stage.rename(TARGET)
    except BaseException:
        backup.rename(TARGET)
        raise
    print(f"INSTALLED={TARGET}\nBACKUP={backup}\nCOUNT=10", flush=True)


def verify(stage):
    stage = Path(stage).resolve()
    rows = json.loads(FIXTURE.read_text())["memories"]
    with sqlite3.connect(f"file:{stage / 'demo-zh.sqlite'}?mode=ro", uri=True) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        records = db.execute("SELECT id, content FROM memories").fetchall()
        assert len(records) == 10
        assert {r[1] for r in records} == {r["text"] for r in rows}
        assert db.execute("SELECT count(DISTINCT memory_id) FROM memory_tags").fetchone()[0] == 10
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(stage / "vectors"))
    try:
        points, cursor = client.scroll("voicemem", limit=100, with_payload=True)
        assert cursor is None and len(points) == 10
        assert {str(p.id) for p in points} == {r[0] for r in records}
        assert {p.payload["data"] for p in points} == {r["text"] for r in rows}
        assert all(p.payload["user_id"] == "voice_user" for p in points)
    finally:
        client.close()
    print("VERIFY=10 facts / 10 vectors / 10 tagged memories / integrity ok", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "install"])
    parser.add_argument("--stage")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif not args.stage:
        parser.error("verify/install requires --stage from a completed prepare")
    elif args.action == "verify":
        verify(args.stage)
    else:
        install(args.stage)
