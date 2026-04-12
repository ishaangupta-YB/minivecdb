"""Edge-case tests for MiniVecDB.

This module stresses unusual but important scenarios:
unicode text, long text, SQL-like content, large top_k, rapid churn,
interleaved search/insert operations, and special-character IDs.

It can be executed both as a standalone script and via pytest.
"""

import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from typing import Callable, Generator, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.embeddings import SimpleEmbeddingEngine
from core.vector_store import VectorStore


def _make_store(storage_dir: str) -> VectorStore:
    """Create a temporary VectorStore configured with SimpleEmbeddingEngine."""
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
    return store


@contextmanager
def _temporary_store(prefix: str) -> Generator[VectorStore, None, None]:
    """Yield a temporary store and always clean up files/resources afterward."""
    storage_dir = tempfile.mkdtemp(prefix=prefix)
    store = None
    try:
        store = _make_store(storage_dir)
        yield store
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(storage_dir, ignore_errors=True)


def _case_unicode_text() -> None:
    """Insert and retrieve unicode text (emoji + Chinese + Arabic)."""
    with _temporary_store("minivecdb_edge_unicode_") as store:
        unicode_text = "Emoji 😀🔥 | Chinese 量子计算 | Arabic مرحبا بالعالم"
        record_id = "unicode_001"
        inserted_id = store.insert(
            text=unicode_text,
            metadata={"category": "unicode", "lang": "mixed"},
            id=record_id,
        )
        assert inserted_id == record_id

        record = store.get(record_id)
        assert record is not None
        assert record.text == unicode_text
        assert record.metadata["category"] == "unicode"


def _case_very_long_text() -> None:
    """Insert and retrieve long text of at least 1000 words."""
    with _temporary_store("minivecdb_edge_long_") as store:
        long_text = " ".join(f"token{i % 120}" for i in range(1200))
        record_id = "long_001"
        store.insert(text=long_text, metadata={"kind": "long"}, id=record_id)

        record = store.get(record_id)
        assert record is not None
        assert len(record.text.split()) >= 1000
        assert record.metadata == {"kind": "long"}


def _case_special_sql_characters() -> None:
    """Ensure SQL-like content is stored safely and does not alter schema state."""
    with _temporary_store("minivecdb_edge_sql_") as store:
        malicious_text = "Robert'); DROP TABLE records; -- \"quoted\"; SELECT * FROM metadata;"
        store.insert(
            text=malicious_text,
            metadata={"category": "security"},
            id="sql_001",
        )

        first_record = store.get("sql_001")
        assert first_record is not None
        assert first_record.text == malicious_text

        # If SQL injection happened, this second insert/count would break.
        store.insert(text="safe follow-up", id="sql_002")
        assert store.count() == 2
        assert store.db.record_exists("sql_002") is True


def _case_empty_results() -> None:
    """Search should return an empty list when filters match no records."""
    with _temporary_store("minivecdb_edge_empty_") as store:
        store.insert(
            text="machine learning and neural networks",
            metadata={"category": "tech"},
            id="empty_001",
        )

        results = store.search(
            query="completely unrelated query",
            top_k=5,
            filters={"category": "missing"},
        )
        assert results == []


def _case_top_k_larger_than_db() -> None:
    """Search should cap results when top_k exceeds total record count."""
    with _temporary_store("minivecdb_edge_topk_") as store:
        store.insert("doc one", id="topk_001")
        store.insert("doc two", id="topk_002")
        store.insert("doc three", id="topk_003")

        results = store.search("doc", top_k=50)
        assert len(results) == 3


def _case_rapid_inserts_and_deletes() -> None:
    """Perform many quick inserts and deletes and verify index consistency."""
    with _temporary_store("minivecdb_edge_rapid_") as store:
        total = 40
        for i in range(total):
            store.insert(
                text=f"rapid insert {i}",
                metadata={"batch": "rapid"},
                id=f"rapid_{i}",
            )

        assert store.count() == total

        for i in range(0, total, 2):
            assert store.delete(f"rapid_{i}") is True

        assert store.count() == total // 2
        for i in range(0, total, 2):
            assert store.get(f"rapid_{i}") is None
        for i in range(1, total, 2):
            assert store.get(f"rapid_{i}") is not None


def _case_concurrent_like_access() -> None:
    """Interleave searches and inserts to simulate concurrent-like usage."""
    with _temporary_store("minivecdb_edge_concurrent_") as store:
        for i in range(8):
            store.insert(text=f"baseline record {i}", id=f"base_{i}")

        for i in range(12):
            results = store.search("baseline", top_k=4)
            assert len(results) > 0
            store.insert(
                text=f"new stream record {i}",
                metadata={"phase": "interleaved"},
                id=f"stream_{i}",
            )

        assert store.count() == 20


def _case_special_character_id() -> None:
    """Insert and retrieve a record whose ID contains special characters."""
    with _temporary_store("minivecdb_edge_id_") as store:
        special_id = "id:special-01_#@!$"
        store.insert(
            text="record with special id",
            metadata={"category": "id-test"},
            id=special_id,
        )

        record = store.get(special_id)
        assert record is not None
        assert record.id == special_id
        assert record.metadata["category"] == "id-test"
        assert store.delete(special_id) is True
        assert store.get(special_id) is None


def run_edge_case_tests(verbose: bool = True) -> Tuple[int, int]:
    """Run all edge-case checks and return (passed, failed) totals."""
    cases: List[Tuple[str, Callable[[], None]]] = [
        ("Unicode text insert/retrieve", _case_unicode_text),
        ("Very long text insert/retrieve", _case_very_long_text),
        ("Special SQL characters safety", _case_special_sql_characters),
        ("Empty results handling", _case_empty_results),
        ("top_k larger than DB", _case_top_k_larger_than_db),
        ("Rapid inserts and deletes", _case_rapid_inserts_and_deletes),
        ("Concurrent-like interleaved access", _case_concurrent_like_access),
        ("Special-character record ID", _case_special_character_id),
    ]

    passed = 0
    failed = 0

    for case_name, case_fn in cases:
        try:
            case_fn()
            passed += 1
            if verbose:
                print(f"PASS: {case_name}")
        except AssertionError as exc:
            failed += 1
            if verbose:
                print(f"FAIL: {case_name} -> {exc}")

    if verbose:
        print("=" * 72)
        print(f"EDGE CASE SUMMARY: {passed} passed, {failed} failed, {passed + failed} total")
        print("=" * 72)

    return passed, failed


def test_edge_case_suite() -> None:
    """Pytest entry point that runs the complete edge-case suite."""
    _passed, failed = run_edge_case_tests(verbose=False)
    assert failed == 0


if __name__ == "__main__":
    """Run edge-case tests as a standalone script with printed summary."""
    _, failed_count = run_edge_case_tests(verbose=True)
    if failed_count > 0:
        raise SystemExit(1)
