from fastapi import APIRouter
from ..config import DB_PATH, EMBEDDING_MODEL_LOCAL_PATH
import os

from ..embeddings import get_model_source, get_model

router = APIRouter()


@router.get("/health")
def health():
    model_ok = False
    model_path = None
    model_source = None
    try:
        model_path = os.path.abspath(EMBEDDING_MODEL_LOCAL_PATH) if EMBEDDING_MODEL_LOCAL_PATH else None
        model_source = get_model_source()
        # Try to ensure model is loadable (best-effort, may trigger download)
        try:
            get_model()
            model_ok = True
        except Exception:
            model_ok = False
    except Exception:
        model_ok = False
    # Try to probe available models (optional) via models_service if present
    models = None
    try:
        from ..services import get_models_cached

        models = None
        try:
            # call synchronously if possible
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models = loop.run_until_complete(get_models_cached())
            loop.close()
        except Exception:
            models = None
    except Exception:
        models = None

    return {"status": "ok", "db": DB_PATH, "model_path": model_path, "model_source": model_source, "model_ok": model_ok, "models_probe": models}


@router.get("/")
def root():
    return {"service": "bartender_memory", "status": "running"}
