# server/embeddings.py
import numpy as np
from .config import EMBEDDING_MODEL, EMBEDDING_MODEL_LOCAL_PATH

# Compatibility shim: some versions of sentence-transformers expect
# `huggingface_hub.cached_download` to exist. Newer huggingface_hub
# exposes `hf_hub_download` instead. Create an alias if needed so
# `sentence_transformers` can import successfully.
try:
    import huggingface_hub as _hf_hub
    if not hasattr(_hf_hub, "cached_download") and hasattr(_hf_hub, "hf_hub_download"):
        # provide a wrapper that accepts the legacy `url` argument that
        # sentence-transformers may pass. If `url` appears to be an HTTP
        # URL, download it directly into a cache file. Otherwise delegate
        # to hf_hub_download mapping sensible kwargs.
        from huggingface_hub import hf_hub_download as _hf_hub_download
        import hashlib, os, tempfile, requests

        def _cached_download(url=None, *args, **kwargs):
            # direct HTTP URL -> download via requests into cache_dir or temp
            if url and (url.startswith("http://") or url.startswith("https://")):
                cache_dir = kwargs.get("cache_dir") or kwargs.get("cache_root") or kwargs.get("cache_folder")
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                else:
                    cache_dir = tempfile.gettempdir()
                # name file deterministically
                name = hashlib.sha1(url.encode("utf-8")).hexdigest()
                dest = os.path.join(cache_dir, name)
                if not os.path.exists(dest) or kwargs.get("force_download"):
                    resp = requests.get(url, stream=True, timeout=30)
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                return dest
            # otherwise assume hub-style download: try to map 'url' to repo_id/filename
            if url and "/" in url and not url.startswith("http"):
                # treat as repo_id or repo_id/filename
                parts = url.split("/")
                repo_id = parts[0]
                filename = "/".join(parts[1:]) if len(parts) > 1 else None
                call_kwargs = {k: v for k, v in kwargs.items()}
                if filename:
                    call_kwargs["filename"] = filename
                call_kwargs.setdefault("repo_id", repo_id)
                return _hf_hub_download(**call_kwargs)
            # fallback: delegate to hf_hub_download with kwargs (may raise)
            return _hf_hub_download(**{k: v for k, v in kwargs.items()})

        _hf_hub.cached_download = _cached_download
except Exception:
    # If importing huggingface_hub fails, we'll let the real import error
    # surface when SentenceTransformer is used so the developer can install
    # the correct packages.
    pass

_model = None

# import SentenceTransformer after the shim above so its imports work
from sentence_transformers import SentenceTransformer


def get_model():
    global _model
    if _model is None:
        # If a local saved sentence-transformers directory exists under
        # EMBEDDING_MODEL_LOCAL_PATH, prefer loading it directly so the
        # server can run fully offline. The typical folder name created by
        # sentence-transformers is `sentence-transformers_<model-name>`.
        try:
            import os as _os
            local_dir = None
            if EMBEDDING_MODEL_LOCAL_PATH:
                candidate = _os.path.join(EMBEDDING_MODEL_LOCAL_PATH, f"sentence-transformers_{EMBEDDING_MODEL}")
                if _os.path.exists(candidate):
                    local_dir = candidate
            if local_dir:
                _model = SentenceTransformer(local_dir)
            else:
                # fall back to model name (may download)
                _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception:
            # let caller see the exception when trying to use the model
            raise
    return _model

# Optional warmup - don't run a blocking warmup at import time in case
# the environment wants to delay heavy downloads. Call get_model() or
# await model warmup at server startup instead.

def embed_text(text):
    """
    Return a numpy array (float32) embedding for the given text.
    """
    model = get_model()
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    # ensure float32 (small)
    return emb.astype(np.float32)

def cosine_sim(a, b):
    # both numpy arrays
    # a: (d,), b: (N, d) or (d,)
    if a.ndim == 1 and b.ndim == 1:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    # vector vs matrix:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    sims = np.dot(b, a)
    return sims  # (N,)