"""Text and message normalization helpers."""
from typing import Any
import json

def extract_text(obj: Any) -> str:
    """Recursively extract text from nested structures (copied from previous main logic)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return str(obj)
    if isinstance(obj, list):
        parts = []
        for it in obj:
            t = extract_text(it)
            if t:
                parts.append(t)
        return "\n".join(parts)
    if isinstance(obj, dict):
        for k in ("text", "content", "message", "body", "utterance"):
            if k in obj:
                return extract_text(obj[k])
        if "parts" in obj and isinstance(obj["parts"], list):
            return extract_text(obj["parts"])
        for v in obj.values():
            t = extract_text(v)
            if t:
                return t
        return ""
    return ""


def guess_role(item: Any) -> str:
    """Try to determine role from an item dict; defaults to 'user'."""
    if not isinstance(item, dict):
        return "user"
    for k in ("role", "author", "from", "sender", "type"):
        v = item.get(k)
        if not v:
            continue
        if isinstance(v, dict):
            role = v.get("role") or v.get("name")
            if role:
                return str(role)
            continue
        sv = str(v).lower()
        if sv in ("user", "system", "assistant", "bot"):
            return "assistant" if sv == "bot" else sv
        if ":" in sv:
            parts = sv.split(":")
            if parts[0] in ("user", "assistant", "system"):
                return parts[0]
    return "user"


def normalize_messages(raw) -> list:
    """Normalize various message shapes into list[dict(role, content)]."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return [{"role": "user", "content": raw}]
        return normalize_messages(parsed)

    if isinstance(raw, dict):
        role = guess_role(raw)
        content = extract_text(raw.get("content") or raw.get("prompt") or raw)
        return [{"role": role, "content": content}]

    if isinstance(raw, list):
        out = []
        for it in raw:
            if isinstance(it, dict):
                role = guess_role(it)
                content_field = it.get("content") or it.get("message") or it.get("prompt") or it.get("text")
                content = extract_text(content_field)
                if not content:
                    content = extract_text(it)
                out.append({"role": role, "content": content})
            else:
                out.append({"role": "user", "content": extract_text(it)})
        return out

    return []
