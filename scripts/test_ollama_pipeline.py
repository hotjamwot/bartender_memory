"""Focused test: insert memory, recall it, forward to Ollama, print raw response.

Usage: python3 scripts/test_ollama_pipeline.py

This script bypasses Cherry Studio and calls the internal pipeline directly.
It will log the outgoing payload, raw response from Ollama, and the mapped OpenAI-like output.
"""
import os
import sys
import asyncio
import logging
import json
import time

# ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("bartender.test_ollama")

# Import internals
try:
    from server.services import remember_text, recall, chat_forward
    from server.core.versioning import map_backend_to_openai
    from server.services import get_models_cached
except Exception as e:
    logger.exception("Failed to import server internals: %s", e)
    raise

async def run_test():
    user_text = f"Test memory for Ollama pipeline at {time.time()}"
    logger.info("Remembering text: %s", user_text)
    try:
        res = await remember_text(user_text, tags=["pipeline-test"])
        logger.info("remember_text returned: %s", res)
    except Exception:
        logger.exception("remember_text failed")
        return

    # Recall the memory
    try:
        rec = await recall(user_text, limit=5)
        logger.info("recall returned: %s", json.dumps(rec, indent=2))
    except Exception:
        logger.exception("recall failed")
        return

    included = rec.get("results", [])
    injected_texts = [f"MEMORY: {it.get('text')}" for it in included]
    if injected_texts:
        system_message = {"role": "system", "content": "\n\n".join(injected_texts)}
        messages_with_memory = [system_message, {"role": "user", "content": user_text}]
    else:
        messages_with_memory = [{"role": "user", "content": user_text}]

    # Log outgoing payload
    logger.info("Outgoing messages_with_memory: %s", messages_with_memory)

    # Try to probe available models
    try:
        models = await get_models_cached()
        logger.info("Models probe: %s", models)
    except Exception:
        logger.exception("Models probe failed (continuing)")

    # Forward to Ollama (use default provider/model behavior in chat_forward)
    try:
        backend_resp = await chat_forward(messages_with_memory, target_provider="ollama", target_model=None, max_tokens=256)
        logger.info("Raw backend_resp: %s", backend_resp)
    except Exception:
        logger.exception("chat_forward failed (did not reach Ollama)")
        return

    # Map to OpenAI-like structure
    try:
        mapped = map_backend_to_openai(backend_resp)
        logger.info("Mapped response: %s", json.dumps(mapped, indent=2))
    except Exception:
        logger.exception("Mapping backend response failed")

if __name__ == "__main__":
    asyncio.run(run_test())
