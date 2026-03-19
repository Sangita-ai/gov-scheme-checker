"""
SQLite database setup and CRUD operations.
Uses Python's built-in sqlite3 module — no extra dependencies.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "schemes.db")


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS schemes (
            id              TEXT PRIMARY KEY,
            name_en         TEXT NOT NULL,
            name_hi         TEXT,
            name_bn         TEXT,
            ministry        TEXT,
            scheme_type     TEXT,
            state           TEXT,
            benefit_type    TEXT,
            benefit_summary TEXT,
            benefit_amount  REAL,
            min_age         INTEGER,
            max_age         INTEGER,
            gender          TEXT,
            income_limit    REAL,
            caste_list      TEXT,
            occupation_list TEXT,
            bpl_required    INTEGER,
            disability_req  INTEGER,
            min_education   TEXT,
            documents       TEXT,
            apply_url       TEXT,
            helpline        TEXT,
            processing_days INTEGER,
            deadline        TEXT,
            rejection_tips  TEXT,
            created_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            profile_json TEXT,
            chat_history TEXT,
            language     TEXT,
            created_at   TEXT,
            updated_at   TEXT
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            scheme_id   TEXT,
            was_helpful INTEGER,
            comment     TEXT,
            created_at  TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Database tables created/verified.")


def upsert_scheme(scheme: dict):
    """Insert or replace a scheme record."""
    conn = get_connection()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        INSERT OR REPLACE INTO schemes
          (id, name_en, name_hi, name_bn, ministry, scheme_type, state,
           benefit_type, benefit_summary, benefit_amount, min_age, max_age,
           gender, income_limit, caste_list, occupation_list, bpl_required,
           disability_req, min_education, documents, apply_url, helpline,
           processing_days, deadline, rejection_tips, created_at)
        VALUES
          (:id, :name_en, :name_hi, :name_bn, :ministry, :scheme_type, :state,
           :benefit_type, :benefit_summary, :benefit_amount, :min_age, :max_age,
           :gender, :income_limit, :caste_list, :occupation_list, :bpl_required,
           :disability_req, :min_education, :documents, :apply_url, :helpline,
           :processing_days, :deadline, :rejection_tips, :created_at)
    """, {**scheme, "created_at": now})
    conn.commit()
    conn.close()


def get_all_schemes() -> list[dict]:
    """Return all schemes as list of dicts."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM schemes").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_scheme_by_id(scheme_id: str) -> Optional[dict]:
    """Return a single scheme by ID."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM schemes WHERE id = ?", (scheme_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_session(session_id: str) -> Optional[dict]:
    """Return session data or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    session = dict(row)
    session["profile_json"] = json.loads(session["profile_json"] or "{}")
    session["chat_history"] = json.loads(session["chat_history"] or "[]")
    return session


def create_session(session_id: str, language: str = "en") -> dict:
    """Create a new empty session."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute("""
        INSERT INTO sessions (session_id, profile_json, chat_history, language, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, "{}", "[]", language, now, now))
    conn.commit()
    conn.close()
    return {
        "session_id": session_id,
        "profile_json": {},
        "chat_history": [],
        "language": language,
    }


def save_session(session_id: str, profile: dict, history: list, language: str):
    """Update an existing session."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute("""
        UPDATE sessions
        SET profile_json = ?, chat_history = ?, language = ?, updated_at = ?
        WHERE session_id = ?
    """, (json.dumps(profile), json.dumps(history), language, now, session_id))
    conn.commit()
    conn.close()



def save_feedback(scheme_id: str, was_helpful: bool, comment: str = ""):
    """Save user feedback for a scheme."""
    now = datetime.utcnow().isoformat()
    conn = get_connection()
    conn.execute("""
        INSERT INTO feedback (scheme_id, was_helpful, comment, created_at)
        VALUES (?, ?, ?, ?)
    """, (scheme_id, 1 if was_helpful else 0, comment, now))
    conn.commit()
    conn.close()