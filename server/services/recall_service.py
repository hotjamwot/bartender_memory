"""Recall logic: find relevant memories for a query."""
import json
import numpy as np
import asyncio
from ..services.db import get_conn
from ..embeddings import embed_text, cosine_sim
from ..core.importance import approx_tokens, compute_recency_bonus, compute_importance
from ..config import SIMILARITY_THRESHOLD, MAX_MEMORIES_PER_RECALL, MAX_TOKENS_PER_RECALL, SIMILARITY_WEIGHT, IMPORTANCE_WEIGHT
import logging
import os

logger = logging.getLogger("bartender")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


async def recall(q: str, limit: int = MAX_MEMORIES_PER_RECALL, min_similarity: float | None = None):
    logger.info("recall called with query: %s", q[:200])
    try:
        db_path = os.path.abspath(os.path.join(BASE_DIR, "data", "memories.db"))
        logger.debug("DB path for recall: %s exists=%s", db_path, os.path.exists(db_path))
    except Exception:
        logger.exception("Error computing DB path for recall")
    q = q.strip()
    q_emb = await asyncio.to_thread(embed_text, q)
    try:
        logger.debug("Query embedding shape: %s dtype=%s", getattr(q_emb, "shape", None), getattr(q_emb, "dtype", None))
    except Exception:
        logger.exception("Failed to log query embedding details")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, text, embedding, tags, times_recalled, created_at, updated_at, pinned, llm_weight FROM memories WHERE deprecated=0")
    rows = cur.fetchall()
    if not rows:
        conn.close()
        logger.debug("No memories found in DB during recall")
        return {"results": []}

    ids = []
    texts = []
    emb_list = []
    meta = []
    logger.debug("Fetched %d memory rows from DB", len(rows))
    for r in rows:
        ids.append(r["id"])
        texts.append(r["text"])
        emb_json = r["embedding"]
        try:
            arr = np.array(json.loads(emb_json), dtype=np.float32)
        except Exception:
            arr = np.zeros_like(q_emb)
        emb_list.append(arr)
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
    try:
        logger.debug("Similarity scores computed (sample): %s", sims.tolist()[:20])
    except Exception:
        logger.debug("Similarity scores computed (non-serializable)")

    pairs = list(zip(ids, texts, sims, meta))
    # Allow a caller to override the similarity threshold for debugging
    threshold_used = float(min_similarity) if min_similarity is not None else float(SIMILARITY_THRESHOLD)
    logger.info("Using similarity threshold=%s (min_similarity override=%s)", threshold_used, min_similarity)
    filtered = [p for p in pairs if float(p[2]) >= threshold_used]
    if not filtered:
        # Log top-k candidates for debugging
        try:
            topk = sorted(pairs, key=lambda x: float(x[2]), reverse=True)[:10]
            debug_list = [
                {"id": int(p[0]), "score": float(p[2]), "text_snippet": (p[1] or "")[:200]} for p in topk
            ]
            logger.debug("Top candidates (none passed threshold): %s", debug_list)
        except Exception:
            logger.exception("Failed to log top candidates")
        conn.close()
        logger.debug("No memories passed similarity threshold=%s", threshold_used)
        return {"results": []}

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

    scored.sort(key=lambda x: x[4], reverse=True)
    selected = [(pid, text, sim, m) for pid, text, sim, imp, combined, m in scored[:limit]]

    total_tokens = sum(approx_tokens(t) for _, t, _, _ in selected)
    while total_tokens > MAX_TOKENS_PER_RECALL and len(selected) > 0:
        pid, text, score, m = selected.pop()
        total_tokens = sum(approx_tokens(t) for _, t, _, _ in selected)

    for pid, _, score, m in selected:
        cur.execute("UPDATE memories SET times_recalled = times_recalled + 1 WHERE id = ?", (pid,))
        cur.execute("SELECT times_recalled, created_at, updated_at, pinned, llm_weight FROM memories WHERE id = ?", (pid,))
        row = cur.fetchone()
        try:
            times_recalled = int(row[0] or 0)
        except Exception:
            times_recalled = 0
        created_at = row[1]
        recency_bonus = compute_recency_bonus(created_at)
        pinned = int(row[3] or 0)
        llm_weight = float(row[4] or 1.0)
        importance = compute_importance(times_recalled, created_at, llm_weight, pinned)
        cur.execute("UPDATE memories SET importance = ? WHERE id = ?", (importance, pid))
    conn.commit()
    results = [{"id": pid, "text": text, "score": float(score)} for pid, text, score, _ in selected]
    logger.info("recall returning %d results (threshold=%s)", len(results), threshold_used)
    conn.close()
    return {"results": results}
