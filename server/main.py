# server/main.py
import os
import json
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from .config import DB_PATH, SIMILARITY_THRESHOLD, DEDUP_SIMILARITY_THRESHOLD, MAX_MEMORIES_PER_RECALL
import asyncio
from .embeddings import embed_text, cosine_sim, get_model
import traceback
import numpy as np

app = FastAPI(title="Bartender Memory (Phase 1)")

# Ensure data directory exists
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        embedding TEXT, -- JSON list
        tags TEXT,
        created_at TEXT,
        updated_at TEXT,
        deprecated INTEGER DEFAULT 0,
        version INTEGER DEFAULT 1,
        previous_id INTEGER DEFAULT NULL,
        times_recalled INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    conn.close()

init_db()


@app.on_event("startup")
async def startup_event():
    # warm up the embedding model in a background thread so the first request isn't slow
    try:
        await asyncio.to_thread(get_model)
    except Exception:
        # don't crash the whole app on warmup failure; the first request will try again
        pass

# Pydantic models
class RememberRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = None

@app.post("/remember")
async def remember(req: RememberRequest):
    """Save a memory. If a very similar memory exists (>= DEDUP_SIMILARITY_THRESHOLD),
    we create a new record that points to previous_id (merge-by-versioning).
    Exceptions are logged to data/error.log for debugging.
    """
    try:
        # main logic follows
        text = req.text.strip()
        tags = ",".join(req.tags) if req.tags else None

        # embed (offload to thread so we don't block the event loop)
        emb = await asyncio.to_thread(embed_text, text)
        emb_list = emb.tolist()

        conn = get_conn()
        cur = conn.cursor()

        # load all embeddings
        cur.execute("SELECT id, embedding, text FROM memories WHERE deprecated=0")
        rows = cur.fetchall()
        best_id = None
        best_sim = -1.0
        if rows:
            db_embs = []
            ids = []
            for r in rows:
                ids.append(r["id"])
                emb_json = r["embedding"]
                try:
                    arr = np.array(json.loads(emb_json), dtype=np.float32)
                except Exception:
                    # fallback if corrupt
                    arr = np.zeros_like(emb)
                db_embs.append(arr)
            db_embs = np.vstack(db_embs)
            sims = cosine_sim(emb, db_embs)  # array of sims
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            best_id = ids[best_idx]

        now = datetime.utcnow().isoformat() + "Z"

        if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
            # merge: create a new record that references the previous (so we keep history)
            # merged text: tack new note onto previous text for clarity
            cur.execute("SELECT text FROM memories WHERE id = ?", (best_id,))
            prev_text = cur.fetchone()["text"]
            merged_text = prev_text + "\n\n[Merged " + now + "]: " + text
            cur.execute("""
                INSERT INTO memories (text, embedding, tags, created_at, updated_at, previous_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (merged_text, json.dumps(emb_list), tags, now, now, best_id, 1))
            new_id = cur.lastrowid
            conn.commit()
            conn.close()
            return {"status":"merged", "new_id": new_id, "previous_id": best_id, "similarity": best_sim}
        else:
            # insert as new
            cur.execute("""
                INSERT INTO memories (text, embedding, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (text, json.dumps(emb_list), tags, now, now))
            new_id = cur.lastrowid
            conn.commit()
            conn.close()
            return {"status":"created", "id": new_id}
    except Exception:
        tb = traceback.format_exc()
        err_path = os.path.join(data_dir, "error.log")
        with open(err_path, "a", encoding="utf-8") as ef:
            ef.write(f"[{datetime.utcnow().isoformat()}] remember error:\n")
            ef.write(tb + "\n")
        raise HTTPException(status_code=500, detail="internal server error")

@app.get("/recall")
async def recall(q: str = Query(..., description="Query text"), limit: int = Query(MAX_MEMORIES_PER_RECALL)):
    """
    Retrieve top-N relevant memories for the query string.
    """
    q = q.strip()
    q_emb = await asyncio.to_thread(embed_text, q)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, tags, times_recalled FROM memories WHERE deprecated=0")
    rows = cur.fetchall()
    if not rows:
        return {"results": []}

    ids = []
    texts = []
    emb_list = []
    meta = []
    for r in rows:
        ids.append(r["id"])
        texts.append(r["text"])
        emb_json = r["embedding"]
        try:
            arr = np.array(json.loads(emb_json), dtype=np.float32)
        except Exception:
            arr = np.zeros_like(q_emb)
        emb_list.append(arr)
        meta.append({"id": r["id"], "tags": r["tags"], "times_recalled": r["times_recalled"]})
    emb_matrix = np.vstack(emb_list)
    sims = cosine_sim(q_emb, emb_matrix)
    # pair and sort
    pairs = list(zip(ids, texts, sims, meta))
    # filter by threshold
    filtered = [p for p in pairs if p[2] >= SIMILARITY_THRESHOLD]
    filtered.sort(key=lambda x: x[2], reverse=True)
    selected = filtered[:limit]
    # increment times_recalled
    for pid, _, score, m in selected:
        cur.execute("UPDATE memories SET times_recalled = times_recalled + 1 WHERE id = ?", (pid,))
    conn.commit()
    conn.close()
    results = [{"id": pid, "text": text, "score": float(score)} for pid, text, score, _ in selected]
    return {"results": results}


@app.get("/health")
def health():
    model_ok = False
    model_path = None
    try:
        from .config import EMBEDDING_MODEL_LOCAL_PATH
        model_path = os.path.abspath(EMBEDDING_MODEL_LOCAL_PATH) if EMBEDDING_MODEL_LOCAL_PATH else None
        if model_path and os.path.exists(model_path):
            model_ok = True
    except Exception:
        model_ok = False
    return {"status": "ok", "db": DB_PATH, "model_path": model_path, "model_ok": model_ok}


@app.get("/")
def root():
    return {"service": "bartender_memory", "status": "running"}

@app.get("/all")
def all_memories(include_deprecated: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    if include_deprecated:
        cur.execute("SELECT * FROM memories ORDER BY updated_at DESC")
    else:
        cur.execute("SELECT * FROM memories WHERE deprecated=0 ORDER BY updated_at DESC")
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"],
            "text": r["text"],
            "tags": r["tags"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "deprecated": bool(r["deprecated"]),
            "version": r["version"],
            "previous_id": r["previous_id"],
            "times_recalled": r["times_recalled"]
        })
    conn.close()
    return {"memories": out}

@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: int, hard: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    if hard:
        cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    else:
        cur.execute("UPDATE memories SET deprecated = 1 WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": memory_id, "hard": bool(hard)}

@app.get("/export")
def export_all():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM memories")
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    conn.close()
    # write to file
    out_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data", "memories_export.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return {"exported_to": out_path}