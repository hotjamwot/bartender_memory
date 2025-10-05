"""Memory creation / update (remember) logic."""
import os
import json
import traceback
import asyncio
from datetime import datetime
from ..services.db import get_conn
from ..config import DEDUP_SIMILARITY_THRESHOLD
from ..embeddings import embed_text, cosine_sim
import numpy as np
import logging

logger = logging.getLogger("bartender")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)


async def remember_text(text: str, tags=None):
    logger.info("remember_text called; db=%s", os.path.abspath(os.path.join(base_dir, "data", "memories.db")))
    try:
        exists = os.path.exists(os.path.join(base_dir, "data", "memories.db"))
        logger.debug("DB exists=%s", exists)
    except Exception:
        logger.exception("Error checking DB path")
    text = text.strip()
    tags_str = ",".join(tags) if tags else None
    pinned = 0
    llm_weight = 1.0
    if tags:
        for t in tags:
            if isinstance(t, str) and t.lower() == "pinned":
                pinned = 1
            if isinstance(t, str) and t.startswith("priority:"):
                p = t.split(":", 1)[1].lower()
                if p == "high":
                    llm_weight = 2.0
                elif p == "low":
                    llm_weight = 0.5

    logger.debug("Computing embedding for text (len=%d)", len(text or ""))
    emb = await asyncio.to_thread(embed_text, text)
    try:
        logger.debug("Embedding shape: %s dtype=%s", getattr(emb, "shape", None), getattr(emb, "dtype", None))
    except Exception:
        logger.exception("Failed to log embedding details")
    emb_list = emb.tolist()

    conn = get_conn()
    cur = conn.cursor()
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
                arr = np.zeros_like(emb)
            db_embs.append(arr)
        db_embs = np.vstack(db_embs)
        sims = cosine_sim(emb, db_embs)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = ids[best_idx]

    now = datetime.utcnow().isoformat() + "Z"

    if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
        cur.execute("SELECT text FROM memories WHERE id = ?", (best_id,))
        prev_text = cur.fetchone()["text"]
        merged_text = prev_text + "\n\n[Merged " + now + "]: " + text
        cur.execute("SELECT version FROM memories WHERE id = ?", (best_id,))
        try:
            prev_version = int(cur.fetchone()["version"] or 1)
        except Exception:
            prev_version = 1
        version = prev_version + 1
        recency_bonus = 1.0
        importance = (0.0 * 0.4) + (recency_bonus * 0.3) + (llm_weight * 0.2) + (pinned * 0.1)
        cur.execute("""
            INSERT INTO memories (text, embedding, tags, created_at, updated_at, previous_id, version, pinned, llm_weight, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (merged_text, json.dumps(emb_list), tags_str, now, now, best_id, version, pinned, llm_weight, importance))
        new_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.info("remember_text merged new_id=%s previous_id=%s similarity=%s", new_id, best_id, best_sim)
        return {"status": "merged", "new_id": new_id, "previous_id": best_id, "similarity": best_sim}
    else:
        recency_bonus = 1.0
        importance = (0.0 * 0.4) + (recency_bonus * 0.3) + (llm_weight * 0.2) + (pinned * 0.1)
        cur.execute("""
            INSERT INTO memories (text, embedding, tags, created_at, updated_at, pinned, llm_weight, importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (text, json.dumps(emb_list), tags_str, now, now, pinned, llm_weight, importance))
        new_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.info("remember_text created id=%s", new_id)
        return {"status": "created", "id": new_id}


async def remember_with_error_handling(text: str, tags=None):
    try:
        return await remember_text(text, tags)
    except Exception:
        tb = traceback.format_exc()
        err_path = os.path.join(data_dir, "error.log")
        with open(err_path, "a", encoding="utf-8") as ef:
            ef.write(f"[{datetime.utcnow().isoformat()}] remember error:\n")
            ef.write(tb + "\n")
        logger.exception("Error in remember_with_error_handling")
        raise
