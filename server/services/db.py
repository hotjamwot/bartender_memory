"""Database helpers for the memories sqlite DB."""
import sqlite3
from ..config import DB_PATH


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        embedding TEXT, -- JSON list
        tags TEXT,
        created_at TEXT,
        updated_at TEXT,
        deprecated INTEGER DEFAULT 0,
        version INTEGER DEFAULT 1,
        previous_id INTEGER DEFAULT NULL,
        times_recalled INTEGER DEFAULT 0,
        importance REAL DEFAULT 0.0,
        pinned INTEGER DEFAULT 0,
        llm_weight REAL DEFAULT 1.0
    )
    """)
    cur.execute("PRAGMA table_info(memories)")
    cols = {r[1] for r in cur.fetchall()}
    if 'importance' not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.0")
    if 'pinned' not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN pinned INTEGER DEFAULT 0")
    if 'llm_weight' not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN llm_weight REAL DEFAULT 1.0")
    conn.commit()
    conn.close()


# initialize on import to preserve previous behavior
init_db()
