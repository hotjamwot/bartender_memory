# server/embeddings.py
import os
import shutil
import logging
import numpy as np
from .config import EMBEDDING_MODEL, EMBEDDING_MODEL_LOCAL_PATH, BASE_DIR

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
logger = logging.getLogger("bartender")
_model_source = None

# import SentenceTransformer after the shim above so its imports work
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore
    logger = logging.getLogger("bartender")
    logger.warning("sentence_transformers not available; falling back to deterministic mock embeddings for local testing")
    import hashlib as _hashlib
    # fallback deterministic embedding generator
    def _mock_embed(text: str):
        import numpy as _np
        h = _hashlib.sha256(text.encode("utf-8")).digest()
        arr = _np.frombuffer(h, dtype=_np.uint8).astype(_np.float32)
        if arr.size < 384:
            arr = _np.pad(arr, (0, 384 - arr.size), mode="wrap")
        else:
            arr = arr[:384]
        arr = arr / (_np.linalg.norm(arr) + 1e-9)
        return arr.astype(_np.float32)


def get_model():
    global _model, _model_source
    if _model is None:
        # Prefer loading a local saved sentence-transformers directory if present.
        # Resolve the configured local path relative to the repository BASE_DIR so
        # loading works regardless of the current working directory.
        # Typical folder: <EMBEDDING_MODEL_LOCAL_PATH>/sentence-transformers_<EMBEDDING_MODEL>
        local_candidate = None
        try:
            resolved_local = EMBEDDING_MODEL_LOCAL_PATH or ""
            # If a relative path was configured, resolve against BASE_DIR
            if resolved_local and not os.path.isabs(resolved_local):
                resolved_local = os.path.abspath(os.path.join(BASE_DIR, resolved_local))
            else:
                resolved_local = os.path.abspath(resolved_local) if resolved_local else ""

            # two possibilities: the configured path already points to the model
            # or it is a parent directory that contains sentence-transformers_<model>
            candidates = []
            if resolved_local:
                candidates.append(os.path.join(resolved_local, f"sentence-transformers_{EMBEDDING_MODEL}"))
                candidates.append(resolved_local)

            chosen = None
            for c in candidates:
                if c and os.path.isdir(c):
                    chosen = c
                    break

            if chosen:
                # Verify model files exist; if not, attempt to download from HF hub
                def _has_required_files(path: str) -> bool:
                    # Basic heuristics: config + model weights + tokenizer
                    cfg = os.path.join(path, "config.json")
                    # model weights: pytorch_model.bin or model.safetensors
                    bin1 = os.path.join(path, "pytorch_model.bin")
                    bin2 = os.path.join(path, "model.safetensors")
                    # tokenizer artifacts
                    tok1 = os.path.join(path, "tokenizer.json")
                    tok2 = os.path.join(path, "vocab.txt")
                    tok3 = os.path.join(path, "tokenizer_config.json")
                    return os.path.exists(cfg) and (os.path.exists(bin1) or os.path.exists(bin2)) and (os.path.exists(tok1) or os.path.exists(tok2) or os.path.exists(tok3))

                if not _has_required_files(chosen):
                    # Try to download the canonical HF repo to this directory
                    try:
                        from huggingface_hub import snapshot_download
                    except Exception as e:
                        raise RuntimeError(
                            f"Local model directory '{chosen}' is missing required files and huggingface_hub is not available to download the model: {e}. "
                            "Install 'huggingface_hub' or place the model at the configured EMBEDDING_MODEL_LOCAL_PATH."
                        ) from e

                    repo_id = f"sentence-transformers/{EMBEDDING_MODEL}"
                    logger.info(f"Model files missing in '{chosen}'; attempting to download {repo_id} from HuggingFace Hub")
                    # Ensure parent exists and temp download location
                    os.makedirs(chosen, exist_ok=True)
                    try:
                        # snapshot_download may return the cache directory where files were written
                        downloaded = snapshot_download(repo_id=repo_id, local_dir=chosen, local_dir_use_symlinks=False)
                        logger.info(f"Downloaded model to '{downloaded}'")
                    except Exception as e:
                        # cleanup partially created dir to avoid leaving a broken folder
                        try:
                            if os.path.isdir(chosen) and not any(os.scandir(chosen)):
                                shutil.rmtree(chosen)
                        except Exception:
                            pass
                        raise RuntimeError(
                            f"Failed to download model '{repo_id}' into '{chosen}': {e}."
                        ) from e

                # After ensuring files, attempt to load
                try:
                    _model = SentenceTransformer(chosen)
                    # record where the model was loaded from
                    _model_source = os.path.abspath(chosen)
                    logger.info(f"Loaded embedding model from local path: {chosen}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load SentenceTransformer from local path '{chosen}': {e}. "
                        "Ensure the directory contains a valid sentence-transformers model (e.g. 'sentence-transformers_all-MiniLM-L6-v2')."
                    ) from e
            else:
                # No local model found; fall back to the model id which allows
                # SentenceTransformer to download from the hub if available.
                try:
                    _model = SentenceTransformer(EMBEDDING_MODEL)
                    _model_source = f"hub:{EMBEDDING_MODEL}"
                    logger.info(f"Loaded embedding model from hub id: {EMBEDDING_MODEL}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load SentenceTransformer model '{EMBEDDING_MODEL}': {e}. "
                        "If you intend to use a local model, set EMBEDDING_MODEL_LOCAL_PATH in config to point at it."
                    ) from e
        except Exception:
            # Propagate exception with stacktrace for callers; callers can catch and log.
            raise
    return _model


def get_model_source() -> str | None:
    """Return a human-readable identifier of where the embedding model was loaded from.

    Examples: absolute path to local folder, or 'hub:all-MiniLM-L6-v2'.
    """
    if _model_source:
        return _model_source
    # if sentence-transformers unavailable, indicate mock
    if SentenceTransformer is None:
        return "mock:MiniLM-fallback"
    return None

# Optional warmup - don't run a blocking warmup at import time in case
# the environment wants to delay heavy downloads. Call get_model() or
# await model warmup at server startup instead.

def embed_text(text):
    """
    Return a numpy array (float32) embedding for the given text.
    """
    # If sentence-transformers is unavailable, use deterministic fallback
    if SentenceTransformer is None:
        out = _mock_embed(text)
        try:
            logger.debug("Using mock embedding shape=%s dtype=%s", getattr(out, "shape", None), getattr(out, "dtype", None))
        except Exception:
            logger.exception("Failed to log mock embedding shape")
        return out

    model = get_model()
    try:
        logger.debug("Using embedding model: %s", get_model_source())
    except Exception:
        logger.exception("Failed to get model source for logging")
    emb = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    # ensure float32 (small)
    out = emb.astype(np.float32)
    try:
        logger.debug("Computed embedding shape=%s dtype=%s", getattr(out, "shape", None), getattr(out, "dtype", None))
    except Exception:
        logger.exception("Failed to log embedding shape")
    return out

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