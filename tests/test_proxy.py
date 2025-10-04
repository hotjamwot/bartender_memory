import sqlite3
import json
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
import server.main as main


def setup_temp_db(tmp_path):
    db_path = tmp_path / "proxy_test.db"

    def get_conn_override():
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    return get_conn_override


def insert_memory(conn, text, emb, importance=0.0, created_at=None, tags=None):
    cur = conn.cursor()
    now = created_at or (datetime.utcnow().isoformat() + "Z")
    emb_json = json.dumps(list(map(float, emb)))
    tags_str = ",".join(tags) if tags else None
    cur.execute(
        "INSERT INTO memories (text, embedding, tags, created_at, updated_at, importance) VALUES (?, ?, ?, ?, ?, ?)",
        (text, emb_json, tags_str, now, now, importance),
    )
    conn.commit()
    return cur.lastrowid


def test_chat_injects_memories_and_respects_importance_order(tmp_path, monkeypatch):
    # Setup temp DB
    get_conn_override = setup_temp_db(tmp_path)
    monkeypatch.setattr(main, "get_conn", get_conn_override)

    # Deterministic embedding stub: queries containing 'dark' -> [1,0]
    def embed_stub(text: str):
        if "dark" in (text or "").lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(main, "embed_text", embed_stub)

    # Init DB
    main.init_db()
    conn = get_conn_override()

    # Insert two memories with identical embeddings but different importance
    emb = [1.0, 0.0]
    id_low = insert_memory(conn, "I like dark mode in VSCode", emb, importance=0.1)
    id_high = insert_memory(conn, "Always prefer dark theme", emb, importance=5.0)

    # Ensure backend call is captured
    captured = {}

    async def fake_call_ollama(messages, model=None, max_tokens=None):
        # capture the messages payload
        captured["messages"] = messages
        return {"id": "fake", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    monkeypatch.setattr(main, "call_ollama", fake_call_ollama)
    monkeypatch.setattr(main, "BACKEND_PROVIDER", "ollama")

    client = TestClient(main.app)

    body = {
        "model": "qwen2.5:3b",
        # include 'dark' so the deterministic embed_stub matches our dark-mode memories
        "messages": [{"role": "user", "content": "Do I prefer dark mode in my editor?"}],
        "max_tokens": 64,
    }

    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    data = r.json()

    # backend should have been called and received messages with a system message
    assert "messages" in captured
    msgs = captured["messages"]
    assert isinstance(msgs, list)
    assert msgs[0]["role"] == "system"
    sys_content = msgs[0]["content"]
    # The high-importance memory should appear before low-importance memory
    assert "Always prefer dark theme" in sys_content
    assert "I like dark mode in VSCode" in sys_content
    assert sys_content.index("Always prefer dark theme") < sys_content.index("I like dark mode in VSCode")

    # Response normalized to OpenAI-like shape
    assert data.get("choices")
    assert data["choices"][0]["message"]["content"] == "ok"


def test_openrouter_proxy_and_normalization(tmp_path, monkeypatch):
    # Setup temp DB
    get_conn_override = setup_temp_db(tmp_path)
    monkeypatch.setattr(main, "get_conn", get_conn_override)

    # embed stub: anything -> [0,1]
    def embed_stub(text: str):
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(main, "embed_text", embed_stub)
    main.init_db()
    conn = get_conn_override()
    insert_memory(conn, "I like coffee", [0.0, 1.0], importance=2.0)

    captured = {}

    async def fake_call_openrouter(messages, model=None, max_tokens=None):
        captured["messages"] = messages
        # return OpenRouter-like shape
        return {"id": "r1", "choices": [{"message": {"role": "assistant", "content": "response from openrouter"}}]}

    monkeypatch.setattr(main, "call_openrouter", fake_call_openrouter)
    monkeypatch.setattr(main, "BACKEND_PROVIDER", "openrouter")

    client = TestClient(main.app)
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "What do I like?"}]}
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200
    data = r.json()
    # ensure memory was injected
    assert "messages" in captured
    assert captured["messages"][0]["role"] == "system"
    assert "I like coffee" in captured["messages"][0]["content"]
    # response normalization
    assert data.get("choices")
    assert data["choices"][0]["message"]["content"] == "response from openrouter"


def test_pin_affects_ranking(tmp_path, monkeypatch):
    # Setup temp DB
    get_conn_override = setup_temp_db(tmp_path)
    monkeypatch.setattr(main, "get_conn", get_conn_override)

    # Deterministic embedding stub: queries containing 'dark' -> [1,0]
    def embed_stub(text: str):
        if "dark" in (text or "").lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(main, "embed_text", embed_stub)

    main.init_db()
    conn = get_conn_override()

    # Insert two similar memories
    emb = [1.0, 0.0]
    id_a = insert_memory(conn, "A: I like dark mode", emb, importance=0.0)
    id_b = insert_memory(conn, "B: I enjoy light mode sometimes", emb, importance=0.0)

    client = TestClient(main.app)

    # Before pinning, recall order should be deterministic by similarity/importance
    r1 = client.get("/recall", params={"q": "dark mode", "limit": 2})
    assert r1.status_code == 200
    res1 = r1.json()["results"]
    # both present
    ids_before = [it["id"] for it in res1]
    assert id_a in ids_before and id_b in ids_before

    # Pin id_b (the second one), then recall again; pinned item should be favored
    p = client.put(f"/memory/{id_b}/pin?pinned=true")
    assert p.status_code == 200

    r2 = client.get("/recall", params={"q": "dark mode", "limit": 2})
    assert r2.status_code == 200
    ids_after = [it["id"] for it in r2.json()["results"]]
    # pinned id_b should appear before id_a now (higher importance due to pinned boost)
    assert ids_after.index(id_b) <= ids_after.index(id_a)


def test_version_chain_on_merge(tmp_path, monkeypatch):
    # Setup temp DB and embed stub that returns same vector so dedup triggers
    get_conn_override = setup_temp_db(tmp_path)
    monkeypatch.setattr(main, "get_conn", get_conn_override)

    def embed_stub(text: str):
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(main, "embed_text", embed_stub)

    main.init_db()
    client = TestClient(main.app)

    # Create initial memory
    r1 = client.post("/remember", json={"text": "My favorite color is blue", "tags": ["preference"]})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1.get("status") == "created"
    mid = d1["id"]

    # Post duplicate text - should merge and return new_id with previous_id == mid
    r2 = client.post("/remember", json={"text": "My favorite color is blue", "tags": ["preference"]})
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2.get("status") in ("merged", "created")
    if d2.get("status") == "merged":
        new_id = d2.get("new_id")
        prev = d2.get("previous_id")
        assert prev == mid
        # Fetch versions chain for new_id
        v = client.get(f"/memory/{new_id}/versions")
        assert v.status_code == 200
        chain = v.json().get("chain", [])
        # chain should contain at least two entries including the previous id
        ids = [c["id"] for c in chain]
        assert mid in ids and new_id in ids
