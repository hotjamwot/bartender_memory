"""Model probing and caching helpers."""
import time
import json
import re
import asyncio
import httpx
import subprocess
import logging
from ..config import BACKEND_PROVIDER, OLLAMA_URL, OPENROUTER_URL, OPENROUTER_API_KEY, MODELS_CACHE_TTL_SECONDS
from ..core.versioning import friendly_label_from_mid

logger = logging.getLogger("bartender")

_models_cache = {"ts": 0, "data": None}
_last_models_probe = {"cli_stdout": None, "cli_stderr": None, "http_body": None, "timestamp": 0}


def _norm(provider: str, mid: str) -> str:
    return f"{provider}-{mid}" if not mid.startswith(f"{provider}-") else mid


async def v1_models():
    provider = BACKEND_PROVIDER
    models = []
    logger.info(f"Probing models for provider={provider} (OLLAMA_URL={OLLAMA_URL}, OPENROUTER_URL={OPENROUTER_URL})")
    async with httpx.AsyncClient(timeout=5.0) as client:
        if provider == "ollama":
            items = []
            try:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["ollama", "list", "--json"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                rc = getattr(proc, "returncode", None)
                stdout = (getattr(proc, "stdout", None) or "")
                stderr = (getattr(proc, "stderr", None) or "")
                logger.debug(f"ollama list returncode={rc}")
                logger.debug(f"ollama list stdout: {stdout}")
                logger.debug(f"ollama list stderr: {stderr}")
                out = stdout.strip() or stderr.strip() or ""
                if out and out[0] in ("[", "{"):
                    try:
                        data = json.loads(out)
                    except Exception:
                        logger.exception("Failed to parse JSON from `ollama list --json` output")
                        data = None
                else:
                    data = None
                if isinstance(data, dict) and data.get("models"):
                    items = data.get("models")
                elif isinstance(data, list):
                    items = data
                else:
                    items = []
            except FileNotFoundError:
                logger.info("`ollama` binary not found on PATH; will try HTTP probe as fallback")
                items = []
            except Exception:
                logger.exception("Error running `ollama list --json`")
                items = []

            if not items:
                try:
                    proc2 = await asyncio.to_thread(
                        subprocess.run,
                        ["ollama", "list"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    stdout2 = (getattr(proc2, "stdout", None) or "")
                    stderr2 = (getattr(proc2, "stderr", None) or "")
                    _last_models_probe["cli_stdout"] = stdout2
                    _last_models_probe["cli_stderr"] = stderr2
                    parsed = []
                    for line in stdout2.splitlines():
                        s = line.strip()
                        if not s:
                            continue
                        low = s.lower()
                        if low.startswith("name") or low.startswith("model") or set(s) <= set("- "):
                            continue
                        parts = s.split()
                        if not parts:
                            continue
                        candidate = parts[0]
                        if not re.match(r"^[A-Za-z0-9_\-:\.]+$", candidate):
                            continue
                        if candidate.lower() in ("name", "image", "model"):
                            continue
                        if candidate not in parsed:
                            parsed.append(candidate)
                    if parsed:
                        items = parsed
                except Exception:
                    logger.exception("Error running plain `ollama list` fallback")
                    items = []

                if not items:
                    tried = []
                    for ep in (f"{OLLAMA_URL}/models", f"{OLLAMA_URL}/api/models"):
                        try:
                            r = await client.get(ep)
                            text = r.text
                            logger.debug(f"Ollama HTTP probe response from {ep}: {text}")
                            _last_models_probe["http_body"] = text
                            if r.status_code == 200:
                                try:
                                    data = r.json()
                                except Exception as e:
                                    logger.warning(f"Failed to parse JSON from Ollama {ep}: {e}; raw={text}")
                                    data = None
                                if isinstance(data, dict) and data.get("models"):
                                    ep_items = data.get("models")
                                elif isinstance(data, list):
                                    ep_items = data
                                else:
                                    ep_items = []
                                if ep_items:
                                    items = ep_items
                                    break
                        except Exception:
                            logger.exception(f"Error probing Ollama endpoint {ep}")
                            tried.append(ep)

            for m in (items or []):
                if isinstance(m, dict):
                    mid_raw = m.get("name") or m.get("id") or m.get("model") or str(m)
                    display = m.get("name") or m.get("description")
                else:
                    mid_raw = str(m)
                    display = None
                mid = _norm(provider, mid_raw)
                models.append({
                    "id": mid,
                    "object": "model",
                    "owned_by": "ollama",
                    "name": display or friendly_label_from_mid(mid, 'ollama')
                })

            if not models:
                fallback = ["qwen2.5:3b"]
                models = [{"id": _norm(provider, m), "object": "model", "owned_by": "ollama", "name": friendly_label_from_mid(_norm(provider, m), 'ollama')} for m in fallback]
            return {"data": models}

        elif provider == "openrouter":
            url = f"{OPENROUTER_URL}/models"
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"} if OPENROUTER_API_KEY else {}
            try:
                r = await client.get(url, headers=headers)
                text = r.text
                logger.debug(f"OpenRouter response from {url}: {text}")
                r.raise_for_status()
                try:
                    data = r.json()
                except Exception as e:
                    logger.warning(f"Failed to parse JSON from OpenRouter {url}: {e}; raw={text}")
                    data = None
                items = data.get("models") if isinstance(data, dict) else data or []
                for m in items:
                    mid_raw = m.get("id") if isinstance(m, dict) else str(m)
                    mid = _norm(provider, mid_raw)
                    models.append({"id": mid, "object": "model", "owned_by": "openrouter", "name": friendly_label_from_mid(mid, 'openrouter')})
                if models:
                    return {"data": models}
            except Exception:
                logger.exception(f"Error probing OpenRouter models at {url}")
                fallback = ["qwen2.5:3b", "gpt-4o-mini", "gpt-4o"]
                models = [{"id": _norm(provider, m), "object": "model", "owned_by": "openrouter", "name": friendly_label_from_mid(_norm(provider, m), 'openrouter')} for m in fallback]
                return {"data": models}

        else:
            fallback = ["qwen2.5:3b", "all-MiniLM-L6-v2"]
            models = [{"id": _norm(provider, m), "object": "model", "owned_by": "bartender"} for m in fallback]
            return {"data": models}


async def get_models_cached():
    now = time.time()
    if _models_cache["data"] and (now - _models_cache["ts"] < MODELS_CACHE_TTL_SECONDS):
        return _models_cache["data"]
    data = await v1_models()
    _models_cache["ts"] = now
    _models_cache["data"] = data
    return data


def get_last_probe():
    return {"last_probe": _last_models_probe}
