"""
+===============================================================+
|  MiniVecDB -- Day 10 Tests: Advanced Metadata Filtering        |
|  File: minivecdb/tests/test_day10.py                           |
|                                                                |
|  Tests the enhanced filter_by_metadata() method that now       |
|  supports:                                                     |
|    1) Exact match:  {"category": "science"}                    |
|    2) List (OR):    {"category": ["science", "tech"]}          |
|    3) $gt / $lt / $gte / $lte  (numeric comparisons)           |
|    4) $ne  (not equal)                                         |
|    5) Combined filters (AND logic across multiple keys)        |
|                                                                |
|  Also tests that VectorStore.search() correctly uses the       |
|  enhanced filters to pre-filter candidates before similarity   |
|  computation.                                                  |
+===============================================================+
"""

import os
import sys
import time
import shutil
import tempfile

import numpy as np

# Add the project root to sys.path so we can import project modules.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.database import DatabaseManager
from core.vector_store import VectorStore
from ARCHITECTURE import generate_id


# ===============================================================
# HELPER: Set up a DatabaseManager with test data
# ===============================================================

def create_test_db() -> tuple:
    """
    Create a temporary SQLite database and populate it with test
    records and metadata for filter testing.

    Returns:
        (db, db_path) tuple.  The caller must call db.close() and
        clean up db_path when done.

    Test data layout:
        rec_001: category=science,  year=2019, status=active
        rec_002: category=tech,     year=2021, status=active
        rec_003: category=science,  year=2023, status=archived
        rec_004: category=history,  year=2020, status=active
        rec_005: category=tech,     year=2025, status=archived
    """
    db_path = os.path.join(tempfile.gettempdir(), f"minivecdb_test_day10_{id(object())}.db")
    db = DatabaseManager(db_path)

    # Insert five records into the default collection.
    records = [
        ("rec_001", "Quantum physics paper",    "default", time.time()),
        ("rec_002", "Machine learning trends",  "default", time.time()),
        ("rec_003", "Biology of cells",         "default", time.time()),
        ("rec_004", "Ancient Rome history",     "default", time.time()),
        ("rec_005", "AI hardware advances",     "default", time.time()),
    ]
    for rec in records:
        db.insert_record(*rec)

    # Attach metadata to each record.
    metadata = [
        ("rec_001", "category", "science"),
        ("rec_001", "year",     "2019"),
        ("rec_001", "status",   "active"),

        ("rec_002", "category", "tech"),
        ("rec_002", "year",     "2021"),
        ("rec_002", "status",   "active"),

        ("rec_003", "category", "science"),
        ("rec_003", "year",     "2023"),
        ("rec_003", "status",   "archived"),

        ("rec_004", "category", "history"),
        ("rec_004", "year",     "2020"),
        ("rec_004", "status",   "active"),

        ("rec_005", "category", "tech"),
        ("rec_005", "year",     "2025"),
        ("rec_005", "status",   "archived"),
    ]
    for record_id, key, value in metadata:
        db.insert_metadata(record_id, key, value)

    return db, db_path


# ===============================================================
# TEST 1: Exact match (existing behaviour, still works)
# ===============================================================

def test_exact_match():
    """Exact string match: {"category": "science"} returns records
    whose category metadata is exactly 'science'."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"category": "science"})
        assert set(result) == {"rec_001", "rec_003"}, (
            f"Expected rec_001 and rec_003, got {result}"
        )
        print("  PASS  test_exact_match")
    finally:
        db.close()
        os.remove(db_path)


def test_exact_match_case_insensitive_key():
    """Metadata key lookups should be case-insensitive and trim-safe."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({" Category ": "science"})
        assert set(result) == {"rec_001", "rec_003"}, (
            f"Expected rec_001 and rec_003, got {result}"
        )
        print("  PASS  test_exact_match_case_insensitive_key")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 2: List match (OR) — match any value in the list
# ===============================================================

def test_list_match():
    """List match: {"category": ["science", "tech"]} returns records
    whose category is 'science' OR 'tech'."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"category": ["science", "tech"]})
        assert set(result) == {"rec_001", "rec_002", "rec_003", "rec_005"}, (
            f"Expected 4 records, got {result}"
        )
        print("  PASS  test_list_match")
    finally:
        db.close()
        os.remove(db_path)


def test_list_match_single_element():
    """A list with one element should behave like an exact match."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"category": ["history"]})
        assert set(result) == {"rec_004"}, (
            f"Expected rec_004, got {result}"
        )
        print("  PASS  test_list_match_single_element")
    finally:
        db.close()
        os.remove(db_path)


def test_list_match_empty_list():
    """An empty list should return no results."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"category": []})
        assert result == [], f"Expected empty list, got {result}"
        print("  PASS  test_list_match_empty_list")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 3: $gt (greater than)
# ===============================================================

def test_gt_operator():
    """$gt: {"year": {"$gt": "2020"}} returns records with year > 2020.
    That's rec_002 (2021), rec_003 (2023), rec_005 (2025)."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"year": {"$gt": "2020"}})
        assert set(result) == {"rec_002", "rec_003", "rec_005"}, (
            f"Expected rec_002, rec_003, rec_005, got {result}"
        )
        print("  PASS  test_gt_operator")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 4: $lt (less than)
# ===============================================================

def test_lt_operator():
    """$lt: {"year": {"$lt": "2021"}} returns records with year < 2021.
    That's rec_001 (2019) and rec_004 (2020)."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"year": {"$lt": "2021"}})
        assert set(result) == {"rec_001", "rec_004"}, (
            f"Expected rec_001, rec_004, got {result}"
        )
        print("  PASS  test_lt_operator")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 5: $gte (greater than or equal)
# ===============================================================

def test_gte_operator():
    """$gte: {"year": {"$gte": "2021"}} returns records with year >= 2021.
    That's rec_002 (2021), rec_003 (2023), rec_005 (2025)."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"year": {"$gte": "2021"}})
        assert set(result) == {"rec_002", "rec_003", "rec_005"}, (
            f"Expected rec_002, rec_003, rec_005, got {result}"
        )
        print("  PASS  test_gte_operator")
    finally:
        db.close()
        os.remove(db_path)


def test_gte_boundary():
    """$gte at exact boundary: {"year": {"$gte": "2019"}} includes
    the boundary value itself. All 5 records have year >= 2019."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"year": {"$gte": "2019"}})
        assert len(result) == 5, f"Expected 5 records, got {len(result)}"
        print("  PASS  test_gte_boundary")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 6: $lte (less than or equal)
# ===============================================================

def test_lte_operator():
    """$lte: {"year": {"$lte": "2020"}} returns records with year <= 2020.
    That's rec_001 (2019) and rec_004 (2020)."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"year": {"$lte": "2020"}})
        assert set(result) == {"rec_001", "rec_004"}, (
            f"Expected rec_001, rec_004, got {result}"
        )
        print("  PASS  test_lte_operator")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 7: $ne (not equal)
# ===============================================================

def test_ne_operator():
    """$ne: {"status": {"$ne": "archived"}} returns records whose
    status is NOT 'archived'. That's rec_001, rec_002, rec_004."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({"status": {"$ne": "archived"}})
        assert set(result) == {"rec_001", "rec_002", "rec_004"}, (
            f"Expected rec_001, rec_002, rec_004, got {result}"
        )
        print("  PASS  test_ne_operator")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 8: Combined filters (AND logic across keys)
# ===============================================================

def test_combined_exact_and_operator():
    """Combined: category=science AND year>2020.
    rec_001 is science/2019 (fails year), rec_003 is science/2023 (passes both).
    Only rec_003 should match."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({
            "category": "science",
            "year": {"$gt": "2020"},
        })
        assert set(result) == {"rec_003"}, (
            f"Expected only rec_003, got {result}"
        )
        print("  PASS  test_combined_exact_and_operator")
    finally:
        db.close()
        os.remove(db_path)


def test_combined_list_and_operator():
    """Combined: category in [science, tech] AND year <= 2021.
    Candidates from list: rec_001(sci/2019), rec_002(tech/2021),
    rec_003(sci/2023), rec_005(tech/2025).
    After year<=2021: rec_001(2019), rec_002(2021)."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({
            "category": ["science", "tech"],
            "year": {"$lte": "2021"},
        })
        assert set(result) == {"rec_001", "rec_002"}, (
            f"Expected rec_001, rec_002, got {result}"
        )
        print("  PASS  test_combined_list_and_operator")
    finally:
        db.close()
        os.remove(db_path)


def test_combined_range_filter():
    """Range: year > 2019 AND year < 2025.
    This uses two separate filter keys... but our metadata table
    stores each key separately. To do a range on the SAME key,
    the user passes a single dict with multiple operators:
    {"year": {"$gt": "2019", "$lt": "2025"}}."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({
            "year": {"$gt": "2019", "$lt": "2025"},
        })
        # 2020, 2021, 2023 match: rec_004, rec_002, rec_003
        assert set(result) == {"rec_002", "rec_003", "rec_004"}, (
            f"Expected rec_002, rec_003, rec_004, got {result}"
        )
        print("  PASS  test_combined_range_filter")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 9: No matches returns empty
# ===============================================================

def test_no_matches():
    """Filters that match nothing should return an empty list."""
    db, db_path = create_test_db()
    try:
        # No record has category=sports
        result = db.filter_by_metadata({"category": "sports"})
        assert result == [], f"Expected [], got {result}"

        # No record has year > 3000
        result = db.filter_by_metadata({"year": {"$gt": "3000"}})
        assert result == [], f"Expected [], got {result}"

        print("  PASS  test_no_matches")
    finally:
        db.close()
        os.remove(db_path)


def test_empty_filters():
    """Empty filters dict should return empty list."""
    db, db_path = create_test_db()
    try:
        result = db.filter_by_metadata({})
        assert result == [], f"Expected [], got {result}"
        print("  PASS  test_empty_filters")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 10: Unknown operator raises ValueError
# ===============================================================

def test_unknown_operator():
    """An unrecognised operator like $regex should raise ValueError."""
    db, db_path = create_test_db()
    try:
        raised = False
        try:
            db.filter_by_metadata({"year": {"$regex": "20.*"}})
        except ValueError as e:
            raised = True
            assert "$regex" in str(e), f"Error should mention $regex: {e}"

        assert raised, "Expected ValueError for unknown operator"
        print("  PASS  test_unknown_operator")
    finally:
        db.close()
        os.remove(db_path)


# ===============================================================
# TEST 11: VectorStore.search() with advanced filters
# ===============================================================

def test_search_with_filters():
    """
    End-to-end test: insert records into VectorStore, then search
    with advanced metadata filters.  Verifies that pre-filtering
    restricts the candidate set correctly and results are ranked.
    """
    storage_path = tempfile.mkdtemp(prefix="minivecdb_test_day10_search_")
    try:
        store = VectorStore(storage_path=storage_path)

        # Insert records with different metadata
        texts_and_meta = [
            ("Quantum computing breakthroughs",        {"category": "science", "year": "2023"}),
            ("Neural network architectures",           {"category": "tech",    "year": "2022"}),
            ("History of the Roman Empire",             {"category": "history", "year": "2019"}),
            ("CRISPR gene editing advances",            {"category": "science", "year": "2021"}),
            ("Deep learning for natural language",      {"category": "tech",    "year": "2024"}),
        ]

        for text, meta in texts_and_meta:
            store.insert(text, metadata=meta)

        # --- Search with exact filter ---
        results = store.search("computing", filters={"category": "science"})
        # Only science records should appear
        for r in results:
            assert r.record.metadata["category"] == "science", (
                f"Expected category=science, got {r.record.metadata}"
            )

        # --- Search with $gt filter ---
        results = store.search("technology", filters={"year": {"$gt": "2022"}})
        for r in results:
            assert int(r.record.metadata["year"]) > 2022, (
                f"Expected year > 2022, got {r.record.metadata['year']}"
            )

        # --- Search with list filter ---
        results = store.search("research", filters={"category": ["science", "tech"]})
        for r in results:
            assert r.record.metadata["category"] in ("science", "tech"), (
                f"Expected science or tech, got {r.record.metadata['category']}"
            )

        # --- Search with combined filters ---
        results = store.search(
            "algorithms",
            filters={"category": "tech", "year": {"$gte": "2023"}},
        )
        for r in results:
            assert r.record.metadata["category"] == "tech", (
                f"Expected tech, got {r.record.metadata['category']}"
            )
            assert int(r.record.metadata["year"]) >= 2023, (
                f"Expected year >= 2023, got {r.record.metadata['year']}"
            )

        # --- Search with filter that matches nothing ---
        results = store.search("anything", filters={"category": "sports"})
        assert results == [], f"Expected empty results, got {len(results)} results"

        # --- Verify results are ranked (scores in order) ---
        results = store.search("neural networks", top_k=3, filters={"category": "tech"})
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].score >= results[i + 1].score, (
                    f"Results not sorted: score {results[i].score} < {results[i+1].score}"
                )
            assert results[0].rank == 1, f"First result rank should be 1, got {results[0].rank}"

        store.close()
        print("  PASS  test_search_with_filters")
    finally:
        shutil.rmtree(storage_path, ignore_errors=True)


# ===============================================================
# RUN ALL TESTS
# ===============================================================

def run_all_tests():
    """Execute every test function and report results."""
    print("=" * 60)
    print("Day 10 Tests: Advanced Metadata Filtering")
    print("=" * 60)

    tests = [
        test_exact_match,
        test_list_match,
        test_list_match_single_element,
        test_list_match_empty_list,
        test_gt_operator,
        test_lt_operator,
        test_gte_operator,
        test_gte_boundary,
        test_lte_operator,
        test_ne_operator,
        test_combined_exact_and_operator,
        test_combined_list_and_operator,
        test_combined_range_filter,
        test_no_matches,
        test_empty_filters,
        test_unknown_operator,
        test_search_with_filters,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL  {test_fn.__name__}: {e}")

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
