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

Starting Bartender (convenience scripts)
---------------------------------------

Two convenience scripts are provided at the repository root to start and stop the server:

- `./start_bartender.sh` — starts the uvicorn server in the background and writes a PID to `.bartender.pid`.
- `./stop_bartender.sh` — stops the server using the PID file and removes it.

Example:

```bash
./start_bartender.sh
# visit http://127.0.0.1:5001
./stop_bartender.sh
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

Using the Shim
--------------

The shim `scripts/shim_cli.py` is a tiny adapter that can either query `/recall` or POST `REMEMBER:` lines to `/remember`.

Examples:

# Recall query (prints JSON results)
```bash
python scripts/shim_cli.py "dark mode"
```

# Post a memory using the REMEMBER prefix
```bash
python scripts/shim_cli.py "REMEMBER: I like dark mode in VSCode"
```

# Post a high-priority memory
```bash
python scripts/shim_cli.py "REMEMBER[high]: My passport is in the drawer"
```

You can also pipe multi-line input; any lines that start with `REMEMBER` will be POSTed:

```bash
cat <<EOF | python scripts/shim_cli.py
REMEMBER: I prefer dark mode
REMEMBER[low]: I'm learning bartending recipes
EOF
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
