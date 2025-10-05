"""Backend proxy helpers for chat completions, embeddings, and model forwarding."""
import asyncio
import json as _json
import logging
try:
    import httpx
except Exception:
    httpx = None
    logging.getLogger("bartender").warning("httpx not available; will use stdlib HTTP fallback for backend calls")
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error
from typing import Any
from ..config import OLLAMA_URL, OPENROUTER_URL, OPENROUTER_API_KEY, BACKEND_PROVIDER, MAX_TOKENS_PER_RECALL
from ..core.normalization import normalize_messages
from ..core.versioning import map_backend_to_openai
from ..embeddings import embed_text, get_model_source
import logging

logger = logging.getLogger("bartender")
DEFAULT_GEN_MODEL = "qwen2.5:3b"


def _stdlib_post_json(url: str, payload: dict, timeout: float = 30.0, headers: dict | None = None):
    headers = headers or {}
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers)
    data = _json.dumps(payload).encode("utf-8")
    req = _urllib_request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with _urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            try:
                return _json.loads(body.decode("utf-8"))
            except Exception:
                return {"raw": body.decode("utf-8", errors="replace")}
    except _urllib_error.HTTPError as e:
        try:
            return _json.loads(e.read().decode("utf-8"))
        except Exception:
            raise
    except Exception:
        raise


from ..config import OLLAMA_WARM_MAX_ATTEMPTS, OLLAMA_WARM_BASE_WAIT


async def ensure_model_ready(model: str, max_attempts: int | None = None, base_wait: float | None = None):
    """Probe Ollama with a tiny generate request to ensure the model is loaded.

    Retries with exponential/backoff up to max_attempts. Logs attempts and returns
    when a non-empty 'response' is observed. Raises after attempts exhausted.
    """
    logger.info("Probing Ollama to ensure model '%s' is ready", model)
    probe_payload = {"model": model, "messages": [{"role": "system", "content": "ping"}], "max_tokens": 1}
    if max_attempts is None:
        max_attempts = int(OLLAMA_WARM_MAX_ATTEMPTS)
    if base_wait is None:
        base_wait = float(OLLAMA_WARM_BASE_WAIT)
    attempt = 0
    last_resp = None
    while attempt < max_attempts:
        attempt += 1
        try:
            if httpx:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(f"{OLLAMA_URL}/api/generate", json=probe_payload)
                    r.raise_for_status()
                    resp = r.json()
            else:
                resp = await asyncio.to_thread(_stdlib_post_json, f"{OLLAMA_URL}/api/generate", probe_payload, 10.0, None)
            logger.debug("ensure_model_ready probe response (attempt %s): %s", attempt, resp)
            last_resp = resp
            # Ollama may use top-level 'response' or 'results'. Consider non-empty strings as ready.
            if isinstance(resp, dict):
                if resp.get("response"):
                    logger.info("Model '%s' ready (response non-empty) on attempt %s", model, attempt)
                    return resp
                # try results array
                res = resp.get("results") or resp.get("outputs") or resp.get("responses")
                if res and isinstance(res, list):
                    # if any element appears to contain text, treat as ready
                    for ritem in res:
                        if isinstance(ritem, dict):
                            content = ritem.get("content") or ritem.get("text") or ritem.get("output")
                            if isinstance(content, str) and content.strip():
                                logger.info("Model '%s' ready (results contain text) on attempt %s", model, attempt)
                                return resp
            # Not ready yet
            wait = base_wait * attempt
            logger.info("Model '%s' not ready yet (attempt %s/%s); waiting %ss before retry", model, attempt, max_attempts, wait)
            await asyncio.sleep(wait)
        except Exception:
            logger.exception("Error probing Ollama for model readiness on attempt %s", attempt)
            await asyncio.sleep(base_wait * attempt)
    logger.warning("ensure_model_ready exhausted %s attempts; last response: %s", max_attempts, last_resp)
    raise RuntimeError(f"model {model} not ready after {max_attempts} attempts")


async def call_ollama(messages, model: str, max_tokens: int | None = None):
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": model, "messages": messages}
    logger.info("Forwarding to Ollama model=%s url=%s", model, url)
    logger.debug("Ollama payload full: %s", payload)
    if httpx:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                resp = r.json()
                logger.debug("Ollama response full: %s", resp)
                # if the model is still loading, Ollama may return an empty 'response'
                # with a done_reason like 'load'. Retry a few times before giving up.
                try_count = 0
                max_retries = 6
                while try_count < max_retries and isinstance(resp, dict) and resp.get("response") == "" and resp.get("done_reason") in ("load",):
                    try_count += 1
                    wait = 1.0 * try_count
                    logger.info("Ollama response indicates model loading; retrying in %ss (attempt %s/%s)", wait, try_count, max_retries)
                    await asyncio.sleep(wait)
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
                    resp = r.json()
                    logger.debug("Ollama retry response full: %s", resp)
                return resp
            except Exception:
                logger.exception("Error calling Ollama via httpx")
                raise
    else:
        # fall back to stdlib urllib in a thread
        try:
            resp = await asyncio.to_thread(_stdlib_post_json, url, payload, 30.0, None)
            logger.debug("Ollama response (stdlib) full: %s", resp)
            # same retry behavior for stdlib fallback
            try_count = 0
            max_retries = 6
            while try_count < max_retries and isinstance(resp, dict) and resp.get("response") == "" and resp.get("done_reason") in ("load",):
                try_count += 1
                wait = 1.0 * try_count
                logger.info("Ollama (stdlib) indicates model loading; retrying in %ss (attempt %s/%s)", wait, try_count, max_retries)
                await asyncio.sleep(wait)
                resp = await asyncio.to_thread(_stdlib_post_json, url, payload, 30.0, None)
                logger.debug("Ollama (stdlib) retry response full: %s", resp)
            return resp
        except Exception:
            logger.exception("Error calling Ollama via stdlib HTTP fallback")
            raise


async def call_openrouter(messages, model: str, max_tokens: int | None = None):
    url = f"{OPENROUTER_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"} if OPENROUTER_API_KEY else {}
    payload = {"model": model, "messages": messages}
    logger.info("Forwarding to OpenRouter model=%s url=%s", model, url)
    logger.debug("OpenRouter payload headers=%s payload=%s", headers, str(payload)[:4000])
    if httpx:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                resp = r.json()
                logger.debug("OpenRouter response: %s", str(resp)[:8000])
                return resp
            except Exception:
                logger.exception("Error calling OpenRouter via httpx")
                raise
    else:
        try:
            resp = await asyncio.to_thread(_stdlib_post_json, url, payload, 30.0, headers)
            logger.debug("OpenRouter response (stdlib): %s", resp)
            return resp
        except Exception:
            logger.exception("Error calling OpenRouter via stdlib HTTP fallback")
            raise


async def embeddings_handler(body: dict):
    inp = body.get("input") or body.get("inputs")
    if not inp:
        raise ValueError("no input provided")
    if isinstance(inp, list):
        # Use MiniLM (embedding model) explicitly for all embedding calculations
        logger.info(f"Computing embeddings using model: {get_model_source()}")
        embs = [await asyncio.to_thread(embed_text, str(i)) for i in inp]
        out = [{"embedding": e.tolist(), "shape": getattr(e, "shape", None)} for e in embs]
        logger.debug("Computed embeddings (list) sample: %s", out[:3])
        return {"data": out}
    logger.info(f"Computing embeddings using model: {get_model_source()}")
    e = await asyncio.to_thread(embed_text, str(inp))
    logger.debug("Computed embedding shape=%s dtype=%s", getattr(e, "shape", None), getattr(e, "dtype", None))
    return {"data": [{"embedding": e.tolist(), "shape": getattr(e, "shape", None)}]}


async def chat_forward(messages_with_memory, target_provider: str, target_model: str | None, max_tokens: int | None):
    # Log which model is being called for generation (exactly what the client requested)
    gen_model = target_model or DEFAULT_GEN_MODEL
    logger.info(f"Forwarding generation request to provider={target_provider} model={gen_model}")
    if target_provider == "ollama":
        # ensure model is warmed/ready before sending the real payload
        logger.info("Ensuring model '%s' is ready before forwarding payload", gen_model)
        await ensure_model_ready(gen_model)
        return await call_ollama(messages_with_memory, model=gen_model, max_tokens=max_tokens)
    elif target_provider == "openrouter":
        return await call_openrouter(messages_with_memory, model=gen_model, max_tokens=max_tokens)
    else:
        # fallback to configured provider
        if BACKEND_PROVIDER == "ollama":
            logger.info("Ensuring model '%s' is ready before forwarding payload (fallback path)", gen_model)
            await ensure_model_ready(gen_model)
            return await call_ollama(messages_with_memory, model=gen_model, max_tokens=max_tokens)
        elif BACKEND_PROVIDER == "openrouter":
            return await call_openrouter(messages_with_memory, model=gen_model, max_tokens=max_tokens)
        else:
            raise RuntimeError(f"unsupported backend: {BACKEND_PROVIDER}")
