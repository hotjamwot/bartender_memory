# server/main.py
import os
import json
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
from .config import DB_PATH, SIMILARITY_THRESHOLD, DEDUP_SIMILARITY_THRESHOLD, MAX_MEMORIES_PER_RECALL, MAX_TOKENS_PER_RECALL, SIMILARITY_WEIGHT, IMPORTANCE_WEIGHT
import asyncio
from .embeddings import embed_text, cosine_sim, get_model
import traceback
import numpy as np

app = FastAPI(title="Bartender Memory (Phase 1)")

# Ensure data directory exists
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model in background thread on startup
    try:
        await asyncio.to_thread(get_model)
    except Exception:
        pass
    yield


app.router.lifespan_context = lifespan

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
        times_recalled INTEGER DEFAULT 0,
        importance REAL DEFAULT 0.0,
        pinned INTEGER DEFAULT 0,
        llm_weight REAL DEFAULT 1.0
    )
    """)
    # Migration-safe: ensure new columns exist if this DB was created earlier
    cur.execute("PRAGMA table_info(memories)")
    cols = {r[1] for r in cur.fetchall()}  # second column is name
    if 'importance' not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.0")
    if 'pinned' not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0")
    if 'llm_weight' not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN llm_weight REAL DEFAULT 1.0")
    conn.commit()
    conn.close()

init_db()


# (startup warmup handled in lifespan above)

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
        # extract simple flags from tags (e.g. priority:high -> pinned/llm_weight)
        pinned = 0
        llm_weight = 1.0
        if req.tags:
            for t in req.tags:
                if isinstance(t, str) and t.lower() == "pinned":
                    pinned = 1
                if isinstance(t, str) and t.startswith("priority:"):
                    # priority:high -> increase weight
                    p = t.split(":", 1)[1].lower()
                    if p == "high":
                        llm_weight = 2.0
                    elif p == "low":
                        llm_weight = 0.5

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
            # determine version number: previous version + 1
            cur.execute("SELECT version FROM memories WHERE id = ?", (best_id,))
            try:
                prev_version = int(cur.fetchone()["version"] or 1)
            except Exception:
                prev_version = 1
            version = prev_version + 1
            # compute a simple importance score. Phase1 uses recall frequency and recency.
            # recency_bonus: newer items get higher recency (in days inverse)
            recency_bonus = 1.0  # default for new merged entry
            importance = (0.0 * 0.4) + (recency_bonus * 0.3) + (llm_weight * 0.2) + (pinned * 0.1)
            cur.execute("""
                INSERT INTO memories (text, embedding, tags, created_at, updated_at, previous_id, version, pinned, llm_weight, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (merged_text, json.dumps(emb_list), tags, now, now, best_id, version, pinned, llm_weight, importance))
            new_id = cur.lastrowid
            conn.commit()
            conn.close()
            return {"status":"merged", "new_id": new_id, "previous_id": best_id, "similarity": best_sim}
        else:
            # insert as new
            # initial importance computation for a fresh memory
            recency_bonus = 1.0
            importance = (0.0 * 0.4) + (recency_bonus * 0.3) + (llm_weight * 0.2) + (pinned * 0.1)
            cur.execute("""
                INSERT INTO memories (text, embedding, tags, created_at, updated_at, pinned, llm_weight, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (text, json.dumps(emb_list), tags, now, now, pinned, llm_weight, importance))
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
    cur.execute("SELECT id, text, embedding, tags, times_recalled, created_at, updated_at, pinned, llm_weight FROM memories WHERE deprecated=0")
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
        # sqlite3.Row doesn't have .get(); access by key and provide defaults
        created_at = r["created_at"] if "created_at" in r.keys() else None
        updated_at = r["updated_at"] if "updated_at" in r.keys() else None
        pinned_val = int(r["pinned"] if "pinned" in r.keys() and r["pinned"] is not None else 0)
        llm_w = float(r["llm_weight"] if "llm_weight" in r.keys() and r["llm_weight"] is not None else 1.0)
        meta.append({
            "id": r["id"],
            "tags": r["tags"],
            "times_recalled": r["times_recalled"],
            "created_at": created_at,
            "updated_at": updated_at,
            "pinned": pinned_val,
            "llm_weight": llm_w
        })
    emb_matrix = np.vstack(emb_list)
    sims = cosine_sim(q_emb, emb_matrix)
    # pair
    pairs = list(zip(ids, texts, sims, meta))
    # filter by similarity threshold first
    filtered = [p for p in pairs if p[2] >= SIMILARITY_THRESHOLD]
    if not filtered:
        return {"results": []}

    # normalize importance across filtered set to [0,1]
    importances = []
    for _, _, _, m in filtered:
        try:
            conn_local = get_conn()
            cur_local = conn_local.cursor()
            cur_local.execute("SELECT importance FROM memories WHERE id = ?", (m["id"],))
            row_imp = cur_local.fetchone()
            imp = float(row_imp[0]) if row_imp and row_imp[0] is not None else 0.0
            importances.append(imp)
            conn_local.close()
        except Exception:
            importances.append(0.0)
    min_imp = min(importances) if importances else 0.0
    max_imp = max(importances) if importances else 1.0
    range_imp = max(1e-9, max_imp - min_imp)

    scored = []
    for (pid, text, sim, m), imp in zip(filtered, importances):
        norm_imp = (imp - min_imp) / range_imp
        combined = SIMILARITY_WEIGHT * sim + IMPORTANCE_WEIGHT * norm_imp
        scored.append((pid, text, sim, imp, combined, m))

    # sort by combined score desc
    scored.sort(key=lambda x: x[4], reverse=True)
    # take top-N by combined score
    selected = [(pid, text, sim, m) for pid, text, sim, imp, combined, m in scored[:limit]]

    # enforce a conservative token budget by approximating tokens from word counts
    # approximate: 1 token ~= 0.75 words -> tokens = words / 0.75
    def approx_tokens(text: str) -> int:
        words = len(text.split()) if text else 0
        return int(words / 0.75)  # conservative (overestimates slightly)

    total_tokens = sum(approx_tokens(t) for _, t, _, _ in selected)
    truncated_by_token_budget = False
    # trim lowest-similarity items until under the token budget
    while total_tokens > MAX_TOKENS_PER_RECALL and len(selected) > 0:
        truncated_by_token_budget = True
        # drop the last (lowest similarity) entry
        pid, text, score, m = selected.pop()
        total_tokens = sum(approx_tokens(t) for _, t, _, _ in selected)

    # increment times_recalled for the final selected set and recompute importance
    for pid, _, score, m in selected:
        cur.execute("UPDATE memories SET times_recalled = times_recalled + 1 WHERE id = ?", (pid,))
        # recompute importance using the formula (phase1 approximated):
        # importance = recall_freq*0.4 + recency_bonus*0.3 + llm_weight*0.2 + pinned*0.1
        cur.execute("SELECT times_recalled, created_at, updated_at, pinned, llm_weight FROM memories WHERE id = ?", (pid,))
        row = cur.fetchone()
        try:
            times_recalled = int(row[0] or 0)
        except Exception:
            times_recalled = 0
        created_at = row[1]
        # compute recency bonus: newer items score higher; recency in days inverse
        recency_bonus = 1.0
        try:
            if created_at:
                created_dt = datetime.fromisoformat(created_at.replace("Z", ""))
                age_days = max((datetime.utcnow() - created_dt).days, 0)
                # more recent => higher bonus (we invert and cap)
                recency_bonus = max(0.1, 1.0 - (age_days / 365.0))
        except Exception:
            recency_bonus = 1.0
        pinned = int(row[3] or 0)
        llm_weight = float(row[4] or 1.0)
        importance = (times_recalled * 0.4) + (recency_bonus * 0.3) + (llm_weight * 0.2) + (pinned * 0.1)
        cur.execute("UPDATE memories SET importance = ? WHERE id = ?", (importance, pid))
    conn.commit()
    # return results
    results = [{"id": pid, "text": text, "score": float(score)} for pid, text, score, _ in selected]
    conn.close()
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


@app.put("/memory/{memory_id}/pin")
def set_pin(memory_id: int, pinned: bool = Query(True, description="Set pinned=true or false")):
    """Pin or unpin a memory. Pinned memories get a small importance boost in scoring."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE memories SET pinned = ? WHERE id = ?", (1 if pinned else 0, memory_id))
    conn.commit()
    conn.close()
    return {"id": memory_id, "pinned": bool(pinned)}


@app.get("/memory/{memory_id}/versions")
def get_versions(memory_id: int):
    """Return the version chain for a memory (walk previous_id links).
    Returns a list ordered from latest -> earliest.
    """
    conn = get_conn()
    cur = conn.cursor()
    chain = []
    cur.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="memory not found")
    # walk back
    cur_row = row
    while cur_row:
        chain.append({k: cur_row[k] for k in cur_row.keys()})
        prev = cur_row["previous_id"]
        if not prev:
            break
        cur.execute("SELECT * FROM memories WHERE id = ?", (prev,))
        cur_row = cur.fetchone()
    conn.close()
    return {"chain": chain}