from fastapi import APIRouter, HTTPException, Query
from ..models.schemas import RememberRequest
from ..services.memory_service import remember_with_error_handling, remember_text
from ..services.recall_service import recall
from ..services.db import get_conn
from ..core.importance import approx_tokens, compute_recency_bonus, compute_importance
from ..services.db import get_conn as _get_conn
from datetime import datetime
import json

router = APIRouter()


@router.post("/remember")
async def remember(req: RememberRequest):
    try:
        return await remember_with_error_handling(req.text, req.tags)
    except Exception:
        raise HTTPException(status_code=500, detail="internal server error")


@router.get("/recall")
async def recall_endpoint(q: str = Query(..., description="Query text"), limit: int = Query(5)):
    return await recall(q=q, limit=limit)


@router.get("/all")
def all_memories(include_deprecated: bool = False):
    conn = _get_conn()
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


@router.delete("/memory/{memory_id}")
def delete_memory(memory_id: int, hard: bool = False):
    conn = _get_conn()
    cur = conn.cursor()
    if hard:
        cur.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    else:
        cur.execute("UPDATE memories SET deprecated = 1 WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted", "id": memory_id, "hard": bool(hard)}


@router.get("/export")
def export_all():
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM memories")
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({k: r[k] for k in r.keys()})
    conn.close()
    out_path = None
    try:
        import os
        out_path = os.path.join(os.path.abspath(os.path.join(__file__, "..", "..")), "data", "memories_export.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    return {"exported_to": out_path}


@router.put("/memory/{memory_id}/pin")
def set_pin(memory_id: int, pinned: bool = Query(True, description="Set pinned=true or false")):
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE memories SET pinned = ? WHERE id = ?", (1 if pinned else 0, memory_id))
    conn.commit()
    conn.close()
    # recompute importance (best-effort)
    try:
        from ..services.memory_service import remember_text
        # reuse recompute by selecting and updating importance manually
        from ..services.db import get_conn as _gc
        c = _gc()
        cur = c.cursor()
        cur.execute("SELECT times_recalled, created_at, pinned, llm_weight FROM memories WHERE id = ?", (memory_id,))
        row = cur.fetchone()
        if row:
            times_recalled = int(row[0] or 0)
            created_at = row[1]
            pinned_v = int(row[2] or 0)
            llm_w = float(row[3] or 1.0)
            imp = compute_importance(times_recalled, created_at, llm_w, pinned_v)
            cur.execute("UPDATE memories SET importance = ? WHERE id = ?", (imp, memory_id))
            c.commit()
        c.close()
    except Exception:
        pass
    return {"id": memory_id, "pinned": bool(pinned)}


@router.get("/memory/{memory_id}/versions")
def get_versions(memory_id: int):
    conn = _get_conn()
    cur = conn.cursor()
    chain = []
    cur.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="memory not found")
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
