# server/main.py
import os
import json
import sqlite3
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
"""FastAPI application assembly for Bartender Memory.

This module creates the FastAPI app, wires routers from the `routes` package,
and sets up a lightweight lifespan that warms up the embedding model in a
background thread on startup (preserving previous behavior).
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from .embeddings import get_model
from .routes import memory_router, proxy_router, health_router
from .services.proxy_service import ensure_model_ready, DEFAULT_GEN_MODEL

app = FastAPI(title="Bartender Memory (Phase 1)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model in background thread on startup (best-effort)
    try:
        await asyncio.to_thread(get_model)
    except Exception:
        pass
    # Warm Ollama default model (best-effort, non-blocking)
    try:
        # run a background task to warm model so startup is not blocked
        async def _warm():
            try:
                await ensure_model_ready(DEFAULT_GEN_MODEL)
            except Exception:
                # log best-effort failure but don't stop the server
                import logging

                logging.getLogger("bartender").warning("Warmup of default model failed at startup (best-effort)")

        asyncio.create_task(_warm())
    except Exception:
        pass
    yield


app.router.lifespan_context = lifespan

# include routers
app.include_router(memory_router)
app.include_router(proxy_router)
app.include_router(health_router)
