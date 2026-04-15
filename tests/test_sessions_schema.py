"""Tests for the v3.0 unified schema: sessions, conversations, messages.

These tests talk to SQLite directly so they don't depend on the full
VectorStore stack (no embedding model to load — keeps the suite fast).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ARCHITECTURE import SCHEMA_SQL, SQL_QUERIES  # noqa: E402
from storage.database import DatabaseManager  # noqa: E402


def _fresh_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    return conn


class TestTriggers(unittest.TestCase):
    """Verify every trigger in the schema fires as specified."""

    def test_trg_create_default_conversation(self):
        conn = _fresh_conn()
        now = time.time()
        cur = conn.execute(
            SQL_QUERIES["insert_session"], ("s1", "/tmp/s1", now, now)
        )
        sid = cur.lastrowid
        conn.commit()

        (count,) = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE session_id = ?", (sid,)
        ).fetchone()
        self.assertEqual(count, 1, "default conversation trigger did not fire")

    def test_trg_create_default_collection(self):
        conn = _fresh_conn()
        now = time.time()
        cur = conn.execute(
            SQL_QUERIES["insert_session"], ("s2", "/tmp/s2", now, now)
        )
        sid = cur.lastrowid
        conn.commit()

        (name,) = conn.execute(
            "SELECT name FROM collections WHERE session_id = ?", (sid,)
        ).fetchone()
        self.assertEqual(name, "default")

    def test_trg_touch_session_on_message(self):
        conn = _fresh_conn()
        now = time.time()
        cur = conn.execute(
            SQL_QUERIES["insert_session"], ("s3", "/tmp/s3", now, now)
        )
        sid = cur.lastrowid
        conn.commit()

        (conv_id,) = conn.execute(
            SQL_QUERIES["get_default_conversation_for_session"], (sid,)
        ).fetchone()

        later = now + 500
        conn.execute(
            SQL_QUERIES["insert_message"],
            (conv_id, "search", "q", "cosine", 5, None, 3, 9.1, None, later),
        )
        conn.commit()

        (touched,) = conn.execute(
            "SELECT last_used_at FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        self.assertAlmostEqual(touched, later, places=3)


class TestCascadeDelete(unittest.TestCase):
    """Every dependent row must disappear when a session row is deleted."""

    def test_deleting_session_cascades_everywhere(self):
        conn = _fresh_conn()
        now = time.time()
        cur = conn.execute(
            SQL_QUERIES["insert_session"], ("scascade", "/tmp/sc", now, now)
        )
        sid = cur.lastrowid
        (conv_id,) = conn.execute(
            SQL_QUERIES["get_default_conversation_for_session"], (sid,)
        ).fetchone()
        (col_id,) = conn.execute(
            SQL_QUERIES["get_collection_id_by_name"], (sid, "default")
        ).fetchone()

        conn.execute(
            SQL_QUERIES["insert_record"],
            ("vec_1", sid, col_id, "hello", now),
        )
        conn.execute(
            SQL_QUERIES["insert_metadata"], ("vec_1", "k", "v")
        )
        conn.execute(
            SQL_QUERIES["insert_message"],
            (conv_id, "search", "q", "cosine", 3, None, 1, 2.0, None, now),
        )
        conn.commit()

        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
        conn.commit()

        for table in ("conversations", "messages", "collections", "records", "metadata"):
            (n,) = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            self.assertEqual(n, 0, f"{table} not cascaded")


class TestAggregateQueries(unittest.TestCase):
    """Session listing and count helper queries; verify the numbers."""

    def test_msg_count_and_record_count_per_session(self):
        conn = _fresh_conn()
        now = time.time()

        # Session A: 2 records, 3 messages.
        cur = conn.execute(
            SQL_QUERIES["insert_session"], ("A", "/A", now, now)
        )
        sid_a = cur.lastrowid
        (conv_a,) = conn.execute(
            SQL_QUERIES["get_default_conversation_for_session"], (sid_a,)
        ).fetchone()
        (col_a,) = conn.execute(
            SQL_QUERIES["get_collection_id_by_name"], (sid_a, "default")
        ).fetchone()
        for i in range(2):
            conn.execute(
                SQL_QUERIES["insert_record"],
                (f"ra_{i}", sid_a, col_a, f"txt{i}", now),
            )
        for i in range(3):
            conn.execute(
                SQL_QUERIES["insert_message"],
                (conv_a, "search", f"q{i}", "cosine", 3, None, 1, 1.0, None, now + i),
            )

        # Session B: 0 records, 1 message.
        cur = conn.execute(
            SQL_QUERIES["insert_session"], ("B", "/B", now, now)
        )
        sid_b = cur.lastrowid
        (conv_b,) = conn.execute(
            SQL_QUERIES["get_default_conversation_for_session"], (sid_b,)
        ).fetchone()
        conn.execute(
            SQL_QUERIES["insert_message"],
            (conv_b, "insert", "some text", None, None, None, 1, 0.5, "vec_x", now),
        )
        conn.commit()

        rows = conn.execute(SQL_QUERIES["list_sessions"]).fetchall()
        by_name = {r[1]: r[0] for r in rows}

        (msg_a,) = conn.execute(
            SQL_QUERIES["count_messages_in_session"],
            (by_name["A"],),
        ).fetchone()
        (rec_a,) = conn.execute(
            SQL_QUERIES["count_records_in_session"],
            (by_name["A"],),
        ).fetchone()
        (msg_b,) = conn.execute(
            SQL_QUERIES["count_messages_in_session"],
            (by_name["B"],),
        ).fetchone()
        (rec_b,) = conn.execute(
            SQL_QUERIES["count_records_in_session"],
            (by_name["B"],),
        ).fetchone()

        self.assertEqual(msg_a, 3)
        self.assertEqual(rec_a, 2)
        self.assertEqual(msg_b, 1)
        self.assertEqual(rec_b, 0)


class TestDatabaseManagerSessionBinding(unittest.TestCase):
    """End-to-end: the wrapper's record ops stay scoped to one session."""

    def test_records_are_isolated_between_sessions(self):
        tmp_dir = tempfile.mkdtemp()
        shared_db = os.path.join(tmp_dir, "shared.db")

        try:
            dm_a = DatabaseManager(
                shared_db, session_name="sess_a", session_storage_path=tmp_dir
            )
            dm_a.insert_record("vec_a", "alpha", "default", time.time())
            dm_a.close()

            dm_b = DatabaseManager(
                shared_db, session_name="sess_b", session_storage_path=tmp_dir
            )
            # Must not see session A's record.
            self.assertFalse(dm_b.record_exists("vec_a"))
            self.assertIsNone(dm_b.get_record("vec_a"))
            self.assertEqual(dm_b.count_records(), 0)

            dm_b.insert_record("vec_b", "beta", "default", time.time())
            self.assertEqual(dm_b.count_records(), 1)
            dm_b.close()

            # Reopen A and verify its record is intact.
            dm_a2 = DatabaseManager(
                shared_db, session_name="sess_a", session_storage_path=tmp_dir
            )
            self.assertEqual(dm_a2.count_records(), 1)
            self.assertTrue(dm_a2.record_exists("vec_a"))
            dm_a2.close()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_log_message_and_get_history(self):
        tmp_dir = tempfile.mkdtemp()
        shared_db = os.path.join(tmp_dir, "shared.db")
        try:
            dm = DatabaseManager(
                shared_db, session_name="hist", session_storage_path=tmp_dir
            )
            dm.log_message(kind="search", query_text="hello",
                           metric="cosine", top_k=5, result_count=2,
                           elapsed_ms=1.2)
            dm.log_message(kind="insert", query_text="new doc",
                           result_count=1, elapsed_ms=0.8,
                           response_ref="vec_new")

            history = dm.get_history(limit=10)
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].kind, "search")
            self.assertEqual(history[0].query_text, "hello")
            self.assertEqual(history[1].response_ref, "vec_new")
            dm.close()
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
