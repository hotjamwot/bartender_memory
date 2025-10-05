"""Services package exposing db, memory, recall and proxy helpers."""
from .db import get_conn, init_db
from .memory_service import remember_text, remember_with_error_handling
from .recall_service import recall
# Do not import models_service at package import time to avoid heavy deps (httpx)


def _lazy_models():
	from . import models_service

	return models_service


def get_models_cached(*args, **kwargs):
	return _lazy_models().get_models_cached(*args, **kwargs)


def v1_models(*args, **kwargs):
	return _lazy_models().v1_models(*args, **kwargs)


def get_last_probe(*args, **kwargs):
	return _lazy_models().get_last_probe(*args, **kwargs)

__all__ = [
	"get_conn",
	"init_db",
	"remember_text",
	"remember_with_error_handling",
	"recall",
	"get_models_cached",
	"v1_models",
	"get_last_probe",
]


# Lazy accessors for proxy_service functions to avoid importing heavy deps at
# package import time (httpx, etc.). Tests or lightweight scripts can still
# import server.services without requiring network/HTTP libraries.
def _lazy_proxy():
	from . import proxy_service

	return proxy_service


def call_ollama(*args, **kwargs):
	return _lazy_proxy().call_ollama(*args, **kwargs)


def call_openrouter(*args, **kwargs):
	return _lazy_proxy().call_openrouter(*args, **kwargs)


def embeddings_handler(*args, **kwargs):
	return _lazy_proxy().embeddings_handler(*args, **kwargs)


def chat_forward(*args, **kwargs):
	return _lazy_proxy().chat_forward(*args, **kwargs)


__all__.extend(["call_ollama", "call_openrouter", "embeddings_handler", "chat_forward"]) 
