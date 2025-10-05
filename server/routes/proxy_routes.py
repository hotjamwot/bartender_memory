from fastapi import APIRouter, HTTPException, Request
import json
from ..services.proxy_service import embeddings_handler, chat_forward
from ..services.models_service import get_models_cached, v1_models, get_last_probe
from ..core.normalization import normalize_messages
from ..core.versioning import map_backend_to_openai, friendly_label_from_mid
from ..services.recall_service import recall
from ..config import BACKEND_PROVIDER, MAX_MEMORIES_PER_RECALL, MAX_TOKENS_PER_RECALL
import logging

router = APIRouter()
logger = logging.getLogger("bartender")


@router.post("/v1/embeddings")
async def v1_embeddings(body: dict):
    try:
        return await embeddings_handler(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="no input provided")


@router.get("/v1/models")
async def v1_models_route():
    return await get_models_cached()


@router.post("/v1/models/refresh")
async def v1_models_refresh():
    # clear cache and return fresh
    from ..services.models_service import _models_cache
    _models_cache["data"] = None
    _models_cache["ts"] = 0
    data = await get_models_cached()
    for item in data.get("data", []):
        if "name" not in item:
            pid = item.get("id") or ""
            owned = item.get("owned_by") or BACKEND_PROVIDER
            item["name"] = friendly_label_from_mid(pid, owned)
    return data


@router.post("/models/refresh")
async def legacy_models_refresh():
    return await v1_models_refresh()


@router.get("/v1/models/debug")
def v1_models_debug():
    return get_last_probe()


@router.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    # Try to parse the incoming request as JSON first. Some frontends (e.g. Cherry)
    # send stringified JSON with Content-Type: text/plain, which FastAPI won't
    # coerce into a dict automatically. Handle that case gracefully here.
    body = None
    try:
        body = await request.json()
    except Exception:
        # Fallback: read raw bytes and attempt to decode/parse
        raw = (await request.body()).decode("utf-8", errors="replace").strip()
        if not raw:
            raise HTTPException(status_code=400, detail="empty request body")
        try:
            body = json.loads(raw)
            logger.warning("Received stringified JSON in request body and parsed it into JSON")
        except Exception as e:
            logger.exception("Failed to json.loads raw body")
        except Exception:
            # Try unwrapping a quoted JSON string (e.g. '"{...}"')
            if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                inner = raw[1:-1]
                try:
                    body = json.loads(inner)
                    logger.warning("Received quoted JSON string in request body and parsed it into JSON")
                except Exception:
                    raise HTTPException(status_code=400, detail="invalid JSON body")
            else:
                raise HTTPException(status_code=400, detail="invalid JSON body")

    # Ensure we have a mapping/dict to work with
    # Log parsed raw body for debugging
    try:
        logger.debug("Parsed request body: %s", json.dumps(body)[:4000])
    except Exception:
        logger.debug("Parsed request body (non-serializable): %s", str(body)[:4000])

    if not isinstance(body, dict):
        # If the parsed JSON is a string (double-encoded), try one more decode
        if isinstance(body, str):
            try:
                parsed = json.loads(body)
                if isinstance(parsed, dict):
                    body = parsed
                    logger.warning("Received JSON where top-level was a string; decoded into dict")
                else:
                    raise HTTPException(status_code=400, detail="invalid JSON body")
            except Exception:
                raise HTTPException(status_code=400, detail="invalid JSON body")
        else:
            raise HTTPException(status_code=400, detail="invalid JSON body")

    try:
        logger.info("/v1/chat/completions parsed body: %s", json.dumps(body)[:2000])
    except Exception:
        logger.info("/v1/chat/completions called (non-serializable body)")

    model = body.get("model")
    logger.info("Requested generation model: %s", model)

    if "messages" not in body:
        prompt = body.get("prompt") or body.get("input") or body.get("inputs")
        if prompt is None:
            body["messages"] = []
        else:
            try:
                body["messages"] = normalize_messages(prompt)
            except Exception:
                logger.exception("Error normalizing prompt into messages")
                raise HTTPException(status_code=400, detail="invalid prompt/messages")
    else:
        try:
            body["messages"] = normalize_messages(body["messages"])
        except Exception:
            logger.exception("Error normalizing messages")
            raise HTTPException(status_code=400, detail="invalid messages")

    target_provider = BACKEND_PROVIDER
    target_model = model
    if isinstance(model, str) and "-" in model:
        possible_prefix, rest = model.split("-", 1)
        if possible_prefix in ("ollama", "openrouter", "bartender"):
            target_provider = possible_prefix
            target_model = rest

    messages = body.get("messages") or []
    try:
        logger.debug("Normalized messages: %s", json.dumps(messages)[:4000])
    except Exception:
        logger.debug("Normalized messages (non-serializable): %s", str(messages)[:4000])
    max_tokens = body.get("max_tokens")
    latest_user = None
    for m in reversed(messages):
        if m.get("role") == "user":
            latest_user = m.get("content")
            break
    if not latest_user:
        raise HTTPException(status_code=400, detail="no user message found in messages")

    recall_limit = min(body.get("n") or body.get("limit") or MAX_MEMORIES_PER_RECALL, MAX_MEMORIES_PER_RECALL)
    # allow a debug override from the client to lower the similarity threshold
    min_sim_override = None
    try:
        dbg = body.get("_bartender_debug_min_similarity")
        if dbg is not None:
            min_sim_override = float(dbg)
    except Exception:
        min_sim_override = None

    logger.info("Calling recall with limit=%s min_similarity_override=%s", recall_limit, min_sim_override)
    recall_resp = await recall(q=latest_user, limit=recall_limit, min_similarity=min_sim_override)
    included = recall_resp.get("results", [])

    injected_texts = []
    tokens_used = 0
    for item in included:
        t = item.get("text")
        tok = len(t.split()) if t else 0
        if tokens_used + tok > MAX_TOKENS_PER_RECALL:
            break
        injected_texts.append(f"MEMORY: {t}")
        tokens_used += tok

    if injected_texts:
        system_message = {"role": "system", "content": "\n\n".join(injected_texts)}
        messages_with_memory = [system_message] + messages
        logger.info("Injected %d memories into prompt", len(injected_texts))
    else:
        messages_with_memory = messages

    try:
        logger.debug("Messages with memory (sample): %s", messages_with_memory[:5])
    except Exception:
        logger.debug("Messages with memory (non-serializable) logged")

    try:
        logger.info("Forwarding to backend provider=%s model=%s; messages_count=%d", target_provider, target_model, len(messages_with_memory))
        backend_resp = await chat_forward(messages_with_memory, target_provider, target_model, max_tokens)
    except Exception as e:
        logger.exception("Error forwarding to backend")
        raise HTTPException(status_code=502, detail=str(e))

    # Log raw backend response for diagnostics
    try:
        logger.debug("Raw backend response: %s", str(backend_resp)[:8000])
    except Exception:
        logger.debug("Raw backend response (non-serializable)")

    mapped = map_backend_to_openai(backend_resp)
    mapped.setdefault("_bartender", {})
    mapped["_bartender"]["memories_injected"] = [{"id": it.get("id"), "score": it.get("score")} for it in included]
    return mapped


@router.get("/models")
async def legacy_models():
    return await get_models_cached()
