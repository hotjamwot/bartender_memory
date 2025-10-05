"""Small helpers for model id friendly labels and mapping backend responses."""
from typing import Any, Dict
import logging

logger = logging.getLogger("bartender")


def friendly_label_from_mid(mid: str, provider: str) -> str:
    core = mid
    prefix = f"{provider}-"
    if core.startswith(prefix):
        core = core[len(prefix):]
    low = core.lower()
    if "qwen" in low:
        label = core.replace(":", " ")
    elif "gpt-4o" in low or "gpt-4" in low:
        label = core.replace("-", " ").replace(":", " ")
    elif "mini" in low or "minilm" in low:
        label = "MiniLM (local)"
    else:
        label = core
    return f"{label} ({provider})"


def map_backend_to_openai(backend_resp: Any) -> Dict:
    # If the backend already uses OpenAI-like 'choices', return it directly
    try:
        logger.debug("Mapping backend response to OpenAI shape; incoming type=%s keys=%s", type(backend_resp), getattr(backend_resp, 'keys', lambda: None)())
    except Exception:
        logger.debug("Mapping backend response to OpenAI shape; incoming repr unavailable")

    if isinstance(backend_resp, dict) and backend_resp.get("choices"):
        return backend_resp

    out = {"id": (backend_resp.get("id") if isinstance(backend_resp, dict) else ""), "object": "chat.completion", "choices": [], "usage": {}}

    # Ollama style: { "results": [ {"content": "..."}, ... ] }
    results = None
    if isinstance(backend_resp, dict):
        results = backend_resp.get("results") or backend_resp.get("outputs") or backend_resp.get("responses")
        # Some Ollama HTTP responses use a top-level 'response' string
        top_response = backend_resp.get("response") if isinstance(backend_resp, dict) else None
        if isinstance(top_response, str) and top_response.strip():
            out["choices"].append({"message": {"role": "assistant", "content": top_response}, "finish_reason": backend_resp.get("done_reason") or "stop"})
            return out

    if results and isinstance(results, list):
        for r in results:
            # r might be a dict or a primitive
            if isinstance(r, dict):
                # common keys: 'content', 'text', 'message', 'output'
                content = r.get("content") or r.get("text") or r.get("output") or r.get("message") or ""
                # 'content' may itself be dict with 'parts'
                if isinstance(content, dict):
                    # try to extract parts
                    if "parts" in content and isinstance(content["parts"], list):
                        content = "".join(map(str, content["parts"]))
                    else:
                        content = str(content)
                role = "assistant"
                out["choices"].append({"message": {"role": role, "content": content}, "finish_reason": r.get("finish_reason") or "stop"})
            else:
                out["choices"].append({"message": {"role": "assistant", "content": str(r)}, "finish_reason": "stop"})
        return out

    # Fallback: try to extract 'choices' in other shapes
    choices = backend_resp.get("choices") if isinstance(backend_resp, dict) else None
    if choices:
        for ch in choices:
            msg = ch.get("message") or ch.get("content") or {}
            if isinstance(msg, dict):
                content = msg.get("content") or ""
                role = msg.get("role", "assistant")
            else:
                content = str(msg)
                role = "assistant"
            out["choices"].append({"message": {"role": role, "content": content}, "finish_reason": ch.get("finish_reason") or "stop"})
        return out

    # As a last resort, try to serialize the whole body
    try:
        out["choices"].append({"message": {"role": "assistant", "content": str(backend_resp)}, "finish_reason": "stop"})
    except Exception:
        logger.exception("Failed to map backend response into a choice; returning empty choices")
    return out
