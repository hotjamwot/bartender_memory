import sqlite3
import numpy as np
from datetime import datetime
from fastapi.testclient import TestClient
import server.main as main


def test_add_recall_merge_and_delete(tmp_path, monkeypatch):
    # Create a temporary sqlite file and monkeypatch get_conn to use it
    db_path = tmp_path / "test.db"

    def get_conn_override():
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(main, "get_conn", get_conn_override)

    # Monkeypatch embed_text to deterministic vectors so we don't load models
    def embed_stub(text: str):
        # text containing 'dark' -> vector A, else vector B
        if "dark" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(main, "embed_text", embed_stub)

    # Initialize DB schema in our temp DB
    main.init_db()

    client = TestClient(main.app)

    # Add a memory
    r = client.post("/remember", json={"text": "I like dark mode in VSCode", "tags": ["preference", "editor"]})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "created"
    mid = data["id"]

    # Recall it
    r2 = client.get("/recall", params={"q": "dark mode", "limit": 3})
    assert r2.status_code == 200
    results = r2.json()["results"]
    assert any("dark mode" in item["text"].lower() for item in results)

    # Add duplicate - should merge (or create new merged entry referencing previous)
    r3 = client.post("/remember", json={"text": "I like dark mode in VSCode", "tags": ["preference", "editor"]})
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3.get("status") in ("merged", "created")
    if d3.get("status") == "merged":
        assert d3.get("previous_id") == mid

    # Soft delete the original memory
    d = client.delete(f"/memory/{mid}")
    assert d.status_code == 200

    # Now recall should not return the deleted id
    r4 = client.get("/recall", params={"q": "dark mode", "limit": 3})
    assert r4.status_code == 200
    res4 = r4.json()["results"]
    assert all(item["id"] != mid for item in res4)


def test_edge_cases(tmp_path, monkeypatch):
    # Setup temp DB and deterministic embeddings
    db_path = tmp_path / "test2.db"

    def get_conn_override():
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(main, "get_conn", get_conn_override)

    def embed_stub(text: str):
        if not text:
            return np.array([0.0, 0.0], dtype=np.float32)
        if "dark" in text.lower():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(main, "embed_text", embed_stub)
    main.init_db()
    client = TestClient(main.app)

    # Empty text should be accepted or rejected deterministically; here we expect created
    r = client.post("/remember", json={"text": "", "tags": []})
    assert r.status_code in (200, 422)

    # Corrupted embedding in DB should not crash recall - simulate by inserting bad JSON
    conn = get_conn_override()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat() + "Z"
    cur.execute("INSERT INTO memories (text, embedding, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                ("bad embedding", "not-a-json", None, now, now))
    conn.commit()
    conn.close()

    r2 = client.get("/recall", params={"q": "anything", "limit": 3})
    assert r2.status_code == 200
