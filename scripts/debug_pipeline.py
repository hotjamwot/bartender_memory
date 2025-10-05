"""Local debug runner to exercise embedding -> remember -> recall -> (optional) generate pipeline.

Run this with the project's venv/python. It calls the internal functions directly so it doesn't need network.
"""
import asyncio
import logging
import json
import os
import sys

# Make sure project root is on sys.path so `server` package can be imported when
# running this script directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from server.embeddings import get_model_source, embed_text  # type: ignore
except Exception:
    # Provide a lightweight fallback for environments without sentence-transformers
    import hashlib
    import numpy as _np

    def get_model_source():
        return "mock:MiniLM-fallback"

    def embed_text(text: str):
        # Deterministic pseudo-embedding: hash -> vector
        h = hashlib.sha256(text.encode("utf-8")).digest()
        arr = _np.frombuffer(h, dtype=_np.uint8).astype(_np.float32)
        # reduce or pad to 384 dims (MiniLM typical dims)
        if arr.size < 384:
            arr = _np.pad(arr, (0, 384 - arr.size), mode="wrap")
        else:
            arr = arr[:384]
        # normalize
        arr = arr / (_np.linalg.norm(arr) + 1e-9)
        return arr.astype(_np.float32)
import importlib
import importlib.util

# Import memory_service and recall_service directly by path to avoid importing
# server.services package __init__ which pulls heavy dependencies.
def _load_module_from_path(fullname, path):
    # fullname should be the fully qualified module name, e.g. 'server.services.memory_service'
    spec = importlib.util.spec_from_file_location(fullname, path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure the package context exists so relative imports inside the module work
    parent = fullname.rpartition(".")[0]
    if parent and parent not in sys.modules:
        # Import the parent package (server.services) which has a lightweight __init__
        importlib.import_module(parent)
    spec.loader.exec_module(mod)
    sys.modules[fullname] = mod
    return mod

base = os.path.abspath(os.path.join(PROJECT_ROOT, "server", "services"))
memory_service = _load_module_from_path("server.services.memory_service", os.path.join(base, "memory_service.py"))
recall_service = _load_module_from_path("server.services.recall_service", os.path.join(base, "recall_service.py"))
proxy_service = None
try:
    proxy_service = _load_module_from_path("server.services.proxy_service", os.path.join(base, "proxy_service.py"))
except Exception:
    proxy_service = None

remember_text = memory_service.remember_text
recall = recall_service.recall
call_ollama = getattr(proxy_service, "call_ollama", None)
DEFAULT_GEN_MODEL = getattr(proxy_service, "DEFAULT_GEN_MODEL", "qwen2.5:3b")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("bartender.debug")

TEST_TEXT = "Today I learned that bartenders appreciate clear instructions."

async def main():
    logger.info("Model source: %s", get_model_source())
    emb = await asyncio.to_thread(embed_text, TEST_TEXT)
    logger.info("Embedding shape=%s dtype=%s sample=%s", getattr(emb, "shape", None), getattr(emb, "dtype", None), emb.tolist()[:6])

    # remember
    res = await remember_text(TEST_TEXT, tags=["test"])
    logger.info("remember_text result: %s", res)

    # recall
    rec = await recall(TEST_TEXT, limit=3)
    logger.info("recall results: %s", json.dumps(rec, indent=2))

    # debug: force min_similarity=0.0 to see available candidates
    try:
        rec_loose = await recall(TEST_TEXT, limit=5, min_similarity=0.0)
        logger.info("recall (min_similarity=0.0) results: %s", json.dumps(rec_loose, indent=2))
    except Exception:
        logger.exception("Error running recall with override")

    # optional: try Ollama generation (will fail if Ollama not running)
    messages = [{"role": "user", "content": TEST_TEXT}]
    try:
        resp = await call_ollama(messages, DEFAULT_GEN_MODEL)
        logger.info("Ollama resp: %s", resp)
    except Exception as e:
        logger.warning("Ollama call failed (is Ollama running?): %s", e)

if __name__ == "__main__":
    asyncio.run(main())
