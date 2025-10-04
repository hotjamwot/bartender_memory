# server/config.py
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "memories.db")

# server port
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5001

# Embedding model (default MiniLM)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_LOCAL_PATH = "./models/MiniLM"

# thresholds
SIMILARITY_THRESHOLD = 0.7
DEDUP_SIMILARITY_THRESHOLD = 0.9
MAX_MEMORIES_PER_RECALL = 5
MAX_TOKENS_PER_RECALL = 512

# Weights for recall ranking: final_score = similarity * SIMILARITY_WEIGHT +
# normalized_importance * IMPORTANCE_WEIGHT
SIMILARITY_WEIGHT = 0.7
IMPORTANCE_WEIGHT = 0.3

# Backend LLM proxy settings
# Supported: 'ollama' (local), 'openrouter' (remote)
BACKEND_PROVIDER = os.environ.get("BARTENDER_BACKEND", "ollama")
# Ollama HTTP endpoint (if using Ollama locally)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# OpenRouter settings (if using OpenRouter)
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://api.openrouter.ai/v1")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")