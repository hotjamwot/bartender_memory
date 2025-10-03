# Bartender Memory — Phase 1 (local)

Quick dev notes and curl examples.

Preload the embedding model (run this once while online):

```bash
source venv/bin/activate
python scripts/preload_model.py
```

Start the server:

```bash
source venv/bin/activate
uvicorn server.main:app --host 127.0.0.1 --port 5001
```

Examples:

Add a memory:

```bash
curl -X POST "http://127.0.0.1:5001/remember" \
  -H "Content-Type: application/json" \
  -d '{"text":"I like dark mode in VSCode","tags":["preference","editor"]}'
```

Recall:

```bash
curl "http://127.0.0.1:5001/recall?q=dark%20mode&limit=3"
```

List all memories:

```bash
curl http://127.0.0.1:5001/all
```

Soft delete:

```bash
curl -X DELETE http://127.0.0.1:5001/memory/1
```

Tests:

```bash
source venv/bin/activate
pytest -q
```
