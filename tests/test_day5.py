"""
+===============================================================+
|  MiniVecDB -- Day 5 Tests                                      |
|  File: minivecdb/tests/test_day5.py                            |
|                                                                |
|  Tests for:                                                    |
|    1. DatabaseManager (storage/database.py)                    |
|    2. VectorStore     (core/vector_store.py)                   |
|                                                                |
|  Run with:                                                     |
|    pytest tests/test_day5.py -v                                |
|    python -m pytest tests/test_day5.py -v                      |
+===============================================================+

WHAT THESE TESTS VERIFY:
    - DatabaseManager creates all tables correctly
    - CRUD on records (insert, get, delete, update, exists)
    - Metadata insert / get / filter / cascade delete
    - Collection create / list / exists / delete
    - VectorStore insert returns a valid ID
    - VectorStore get returns a VectorRecord with correct fields
    - VectorStore insert_batch inserts multiple records
    - VectorStore count returns correct numbers
    - Duplicate ID insertion raises ValueError
"""

import os
import sys
import time

import numpy as np
import pytest

# ---------------------------------------------------------------
# Make sure Python can find our project modules.
# We add the project root (one directory above tests/) to sys.path.
# ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ARCHITECTURE import VectorRecord
from storage.database import DatabaseManager
from core.vector_store import VectorStore


# ===============================================================
# FIXTURES -- Reusable test setup / teardown
# ===============================================================
#
# pytest fixtures let us create temporary databases and directories
# that are automatically cleaned up after each test.  The "yield"
# keyword splits setup (before yield) from teardown (after yield).
#
# ===============================================================


@pytest.fixture
def tmp_db(tmp_path):
    """
    Create a temporary DatabaseManager backed by a temp SQLite file.

    tmp_path is a built-in pytest fixture that gives us a unique
    temporary directory for each test.  After the test, pytest
    automatically deletes it.
    """
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)
    yield db
    db.close()


@pytest.fixture
def tmp_store(tmp_path):
    """
    Create a temporary VectorStore in a temp directory.

    This uses SimpleEmbeddingEngine (bag-of-words fallback) since
    we don't want to download a 80 MB model in tests.
    """
    storage_dir = str(tmp_path / "vecstore")
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    yield store
    store.db.close()


# ===============================================================
# TESTS: DatabaseManager -- Table creation
# ===============================================================

class TestDatabaseManagerSetup:
    """Verify that DatabaseManager creates tables correctly."""

    def test_tables_created(self, tmp_db):
        """The three core tables should exist after init."""
        # sqlite_master is SQLite's internal catalog of all objects.
        cursor = tmp_db._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "collections" in tables
        assert "records" in tables
        assert "metadata" in tables

    def test_default_collection_exists(self, tmp_db):
        """The 'default' collection should be auto-created by SCHEMA_SQL."""
        assert tmp_db.collection_exists("default") is True

    def test_foreign_keys_enabled(self, tmp_db):
        """PRAGMA foreign_keys should be ON."""
        cursor = tmp_db._conn.execute("PRAGMA foreign_keys")
        # Returns (1,) if ON, (0,) if OFF.
        assert cursor.fetchone()[0] == 1


# ===============================================================
# TESTS: DatabaseManager -- Record CRUD
# ===============================================================

class TestRecordCRUD:
    """Test insert, get, update, delete, and exists for records."""

    def test_insert_and_get_record(self, tmp_db):
        """insert_record + get_record should round-trip correctly."""
        ts = time.time()
        tmp_db.insert_record("vec_001", "Hello world", "default", ts)
        row = tmp_db.get_record("vec_001")

        # row is (id, text, collection, created_at)
        assert row is not None
        assert row[0] == "vec_001"
        assert row[1] == "Hello world"
        assert row[2] == "default"
        assert row[3] == pytest.approx(ts, abs=0.01)

    def test_get_nonexistent_record_returns_none(self, tmp_db):
        """Getting an ID that doesn't exist should return None."""
        assert tmp_db.get_record("does_not_exist") is None

    def test_record_exists_true(self, tmp_db):
        """record_exists should return True for an existing record."""
        tmp_db.insert_record("vec_002", "Test", "default", time.time())
        assert tmp_db.record_exists("vec_002") is True

    def test_record_exists_false(self, tmp_db):
        """record_exists should return False for a missing ID."""
        assert tmp_db.record_exists("ghost") is False

    def test_delete_record_success(self, tmp_db):
        """delete_record should return True and remove the record."""
        tmp_db.insert_record("vec_003", "Delete me", "default", time.time())
        assert tmp_db.delete_record("vec_003") is True
        assert tmp_db.get_record("vec_003") is None

    def test_delete_record_nonexistent(self, tmp_db):
        """delete_record should return False if ID doesn't exist."""
        assert tmp_db.delete_record("nope") is False

    def test_update_record_text(self, tmp_db):
        """update_record_text should change the text field."""
        tmp_db.insert_record("vec_004", "Old text", "default", time.time())
        result = tmp_db.update_record_text("vec_004", "New text")
        assert result is True
        row = tmp_db.get_record("vec_004")
        assert row[1] == "New text"

    def test_update_nonexistent_returns_false(self, tmp_db):
        """update_record_text should return False for a missing ID."""
        assert tmp_db.update_record_text("ghost", "new") is False


# ===============================================================
# TESTS: DatabaseManager -- Metadata
# ===============================================================

class TestMetadata:
    """Test metadata insert, get, delete, and filter."""

    def test_insert_and_get_metadata(self, tmp_db):
        """Metadata should round-trip as a dict."""
        tmp_db.insert_record("vec_m1", "Meta test", "default", time.time())
        tmp_db.insert_metadata("vec_m1", "category", "science")
        tmp_db.insert_metadata("vec_m1", "year", "2024")

        meta = tmp_db.get_metadata("vec_m1")
        assert meta == {"category": "science", "year": "2024"}

    def test_get_metadata_empty(self, tmp_db):
        """A record with no metadata should return an empty dict."""
        tmp_db.insert_record("vec_m2", "No meta", "default", time.time())
        assert tmp_db.get_metadata("vec_m2") == {}

    def test_delete_metadata(self, tmp_db):
        """delete_metadata should remove all tags for a record."""
        tmp_db.insert_record("vec_m3", "Del meta", "default", time.time())
        tmp_db.insert_metadata("vec_m3", "k1", "v1")
        tmp_db.insert_metadata("vec_m3", "k2", "v2")

        tmp_db.delete_metadata("vec_m3")
        assert tmp_db.get_metadata("vec_m3") == {}

    def test_cascade_delete_removes_metadata(self, tmp_db):
        """Deleting a record should auto-delete its metadata (CASCADE)."""
        tmp_db.insert_record("vec_m4", "Cascade", "default", time.time())
        tmp_db.insert_metadata("vec_m4", "tag", "important")

        # Verify metadata exists before delete
        assert tmp_db.get_metadata("vec_m4") == {"tag": "important"}

        # Delete the RECORD -- metadata should vanish via CASCADE
        tmp_db.delete_record("vec_m4")
        assert tmp_db.get_metadata("vec_m4") == {}

    def test_filter_by_metadata_single(self, tmp_db):
        """filter_by_metadata with one filter should return matching IDs."""
        tmp_db.insert_record("vec_f1", "Rec 1", "default", time.time())
        tmp_db.insert_record("vec_f2", "Rec 2", "default", time.time())
        tmp_db.insert_metadata("vec_f1", "color", "red")
        tmp_db.insert_metadata("vec_f2", "color", "blue")

        result = tmp_db.filter_by_metadata({"color": "red"})
        assert result == ["vec_f1"]

    def test_filter_by_metadata_intersection(self, tmp_db):
        """Multiple filters should AND together (intersection)."""
        tmp_db.insert_record("vec_f3", "Rec 3", "default", time.time())
        tmp_db.insert_record("vec_f4", "Rec 4", "default", time.time())
        tmp_db.insert_metadata("vec_f3", "color", "red")
        tmp_db.insert_metadata("vec_f3", "size", "large")
        tmp_db.insert_metadata("vec_f4", "color", "red")
        tmp_db.insert_metadata("vec_f4", "size", "small")

        # Only vec_f3 is red AND large
        result = tmp_db.filter_by_metadata({"color": "red", "size": "large"})
        assert result == ["vec_f3"]

    def test_filter_empty_returns_empty(self, tmp_db):
        """Filtering with an empty dict should return empty list."""
        assert tmp_db.filter_by_metadata({}) == []


# ===============================================================
# TESTS: DatabaseManager -- Collections
# ===============================================================

class TestCollections:
    """Test collection CRUD operations."""

    def test_create_collection(self, tmp_db):
        """Creating a new collection should make it visible."""
        tmp_db.create_collection("science", 384, "Science papers")
        assert tmp_db.collection_exists("science") is True

    def test_list_collections_includes_default(self, tmp_db):
        """list_collections should include the auto-created 'default'."""
        cols = tmp_db.list_collections()
        names = [c[0] for c in cols]
        assert "default" in names

    def test_delete_collection(self, tmp_db):
        """Deleting a collection should remove it."""
        tmp_db.create_collection("temp", 384, "Temporary")
        assert tmp_db.delete_collection("temp") is True
        assert tmp_db.collection_exists("temp") is False

    def test_collection_exists_false(self, tmp_db):
        """collection_exists returns False for non-existent collections."""
        assert tmp_db.collection_exists("imaginary") is False


# ===============================================================
# TESTS: DatabaseManager -- Statistics
# ===============================================================

class TestStatistics:
    """Test count_records and stats_per_collection."""

    def test_count_records_all(self, tmp_db):
        """count_records(None) should count all records."""
        assert tmp_db.count_records() == 0
        tmp_db.insert_record("r1", "A", "default", time.time())
        tmp_db.insert_record("r2", "B", "default", time.time())
        assert tmp_db.count_records() == 2

    def test_count_records_by_collection(self, tmp_db):
        """count_records(collection) should filter correctly."""
        tmp_db.create_collection("other")
        tmp_db.insert_record("r3", "A", "default", time.time())
        tmp_db.insert_record("r4", "B", "other", time.time())
        assert tmp_db.count_records("default") == 1
        assert tmp_db.count_records("other") == 1

    def test_stats_per_collection(self, tmp_db):
        """stats_per_collection should return counts per collection."""
        tmp_db.create_collection("cats")
        tmp_db.insert_record("r5", "X", "default", time.time())
        tmp_db.insert_record("r6", "Y", "cats", time.time())
        tmp_db.insert_record("r7", "Z", "cats", time.time())

        stats = tmp_db.stats_per_collection()
        assert stats["default"] == 1
        assert stats["cats"] == 2


# ===============================================================
# TESTS: DatabaseManager -- Record ID retrieval
# ===============================================================

class TestRecordIDRetrieval:
    """Test get_all_record_ids and get_record_ids_in_collection."""

    def test_get_all_record_ids(self, tmp_db):
        """get_all_record_ids should return every ID ordered by time."""
        t = time.time()
        tmp_db.insert_record("r_a", "A", "default", t)
        tmp_db.insert_record("r_b", "B", "default", t + 1)
        ids = tmp_db.get_all_record_ids()
        assert ids == ["r_a", "r_b"]

    def test_get_record_ids_in_collection(self, tmp_db):
        """Should only return IDs from the specified collection."""
        t = time.time()
        tmp_db.create_collection("col1")
        tmp_db.insert_record("x1", "A", "default", t)
        tmp_db.insert_record("x2", "B", "col1", t + 1)
        assert tmp_db.get_record_ids_in_collection("col1") == ["x2"]


# ===============================================================
# TESTS: VectorStore -- Insert & Get
# ===============================================================

class TestVectorStoreInsert:
    """Test VectorStore insert and get operations."""

    def test_insert_returns_valid_id(self, tmp_store):
        """insert() should return a string ID starting with 'vec_'."""
        rid = tmp_store.insert("The cat sat on the mat")
        assert isinstance(rid, str)
        assert rid.startswith("vec_")

    def test_get_returns_vector_record(self, tmp_store):
        """get() should return a VectorRecord with correct fields."""
        rid = tmp_store.insert(
            "Python is great",
            metadata={"topic": "programming"},
        )
        record = tmp_store.get(rid)

        # Verify it's a VectorRecord instance
        assert isinstance(record, VectorRecord)

        # Verify all fields
        assert record.id == rid
        assert record.text == "Python is great"
        assert record.metadata == {"topic": "programming"}
        assert record.collection == "default"
        assert isinstance(record.created_at, float)

        # Verify vector shape and type
        assert record.vector.shape == (384,)
        assert record.vector.dtype == np.float32

    def test_get_nonexistent_returns_none(self, tmp_store):
        """get() with a non-existent ID should return None."""
        assert tmp_store.get("does_not_exist") is None

    def test_insert_with_custom_id(self, tmp_store):
        """insert() should accept a custom ID."""
        rid = tmp_store.insert("Custom ID test", id="my_custom_id")
        assert rid == "my_custom_id"
        record = tmp_store.get("my_custom_id")
        assert record is not None
        assert record.text == "Custom ID test"

    def test_duplicate_id_raises_valueerror(self, tmp_store):
        """Inserting with a duplicate ID should raise ValueError."""
        tmp_store.insert("First", id="dup_001")
        with pytest.raises(ValueError, match="already exists"):
            tmp_store.insert("Second", id="dup_001")

    def test_insert_with_metadata(self, tmp_store):
        """Metadata should be stored and retrievable."""
        rid = tmp_store.insert(
            "Metadata test",
            metadata={"author": "Alice", "year": "2024"},
        )
        record = tmp_store.get(rid)
        assert record.metadata == {"author": "Alice", "year": "2024"}


# ===============================================================
# TESTS: VectorStore -- Batch Insert
# ===============================================================

class TestVectorStoreBatch:
    """Test VectorStore insert_batch."""

    def test_insert_batch_returns_ids(self, tmp_store):
        """insert_batch should return a list of IDs."""
        ids = tmp_store.insert_batch(
            texts=["Hello", "World", "Foo"],
        )
        assert len(ids) == 3
        for rid in ids:
            assert isinstance(rid, str)
            assert rid.startswith("vec_")

    def test_insert_batch_records_retrievable(self, tmp_store):
        """Each batch-inserted record should be retrievable via get()."""
        texts = ["Alpha", "Beta", "Gamma"]
        metadata_list = [
            {"idx": "0"},
            {"idx": "1"},
            {"idx": "2"},
        ]
        ids = tmp_store.insert_batch(texts=texts, metadata_list=metadata_list)

        for i, rid in enumerate(ids):
            record = tmp_store.get(rid)
            assert record is not None
            assert record.text == texts[i]
            assert record.metadata == metadata_list[i]

    def test_insert_batch_duplicate_id_raises(self, tmp_store):
        """Batch insert with a pre-existing ID should raise ValueError."""
        tmp_store.insert("Existing", id="batch_dup")
        with pytest.raises(ValueError, match="already exists"):
            tmp_store.insert_batch(
                texts=["New"],
                ids=["batch_dup"],
            )

    def test_insert_batch_duplicate_ids_in_same_call_raises(self, tmp_store):
        """Batch insert should reject duplicate IDs in the same request."""
        with pytest.raises(ValueError, match="Duplicate IDs"):
            tmp_store.insert_batch(
                texts=["First", "Second"],
                ids=["dup_same", "dup_same"],
            )
        assert tmp_store.count() == 0

    def test_insert_batch_save_failure_compensates_sqlite(self, tmp_store, monkeypatch):
        """If save fails, batch insert should remove previously inserted SQLite rows."""

        def fail_save() -> None:
            raise OSError("Simulated disk write failure")

        monkeypatch.setattr(tmp_store, "save", fail_save)

        with pytest.raises(RuntimeError, match="Batch insert failed while saving"):
            tmp_store.insert_batch(texts=["A", "B"])

        assert tmp_store.count() == 0


# ===============================================================
# TESTS: VectorStore -- Count
# ===============================================================

class TestVectorStoreCount:
    """Test VectorStore count method."""

    def test_count_starts_at_zero(self, tmp_store):
        """A fresh store should have 0 records."""
        assert tmp_store.count() == 0

    def test_count_after_inserts(self, tmp_store):
        """count() should match the number of inserted records."""
        tmp_store.insert("One")
        tmp_store.insert("Two")
        tmp_store.insert("Three")
        assert tmp_store.count() == 3

    def test_count_by_collection(self, tmp_store):
        """count(collection) should only count that collection."""
        tmp_store.db.create_collection("special")
        tmp_store.insert("Default record")
        tmp_store.insert("Special record", collection="special")

        assert tmp_store.count("default") == 1
        assert tmp_store.count("special") == 1
        assert tmp_store.count() == 2


# ===============================================================
# TESTS: VectorStore -- Persistence (save / load)
# ===============================================================

class TestVectorStorePersistence:
    """Test that vectors survive a save/load cycle."""

    def test_save_creates_files(self, tmp_store):
        """After insert, vectors.npy and id_mapping.json should exist."""
        tmp_store.insert("Persistence test")
        vectors_path = os.path.join(tmp_store.storage_path, "vectors.npy")
        mapping_path = os.path.join(
            tmp_store.storage_path, "id_mapping.json"
        )
        assert os.path.exists(vectors_path)
        assert os.path.exists(mapping_path)

    def test_reload_preserves_vectors(self, tmp_path):
        """
        Creating a new VectorStore on the same directory should
        reload previously saved vectors.
        """
        storage_dir = str(tmp_path / "reload_test")

        # Create store and insert a record
        store1 = VectorStore(storage_path=storage_dir)
        rid = store1.insert("Reload me", id="reload_001")
        store1.db.close()

        # Create a NEW store pointing at the same directory
        store2 = VectorStore(storage_path=storage_dir)

        # The record should still be there
        record = store2.get("reload_001")
        assert record is not None
        assert record.text == "Reload me"
        assert record.vector.shape == (384,)
        store2.db.close()


class TestVectorStoreIntegrity:
    """Test fail-fast behavior when vector mapping is corrupted."""

    def test_get_raises_on_missing_id_mapping(self, tmp_store):
        """get() should raise RuntimeError if SQLite row has no vector mapping."""
        rid = tmp_store.insert("Corruption check")

        # Simulate in-memory corruption by dropping the mapping entry.
        tmp_store._id_list = []
        tmp_store._rebuild_id_index()

        with pytest.raises(RuntimeError, match="Data integrity error"):
            tmp_store.get(rid)
