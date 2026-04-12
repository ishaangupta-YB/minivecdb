"""
+===============================================================+
|  MiniVecDB -- Day 7 Tests                                      |
|  File: minivecdb/tests/test_day7.py                            |
|                                                                |
|  Tests for CRUD operations and collection management:          |
|    1. delete()            -- remove a record                   |
|    2. update()            -- modify text and/or metadata       |
|    3. create_collection() -- create a new collection           |
|    4. list_collections()  -- list all collections              |
|    5. delete_collection() -- remove a collection + its records |
|    6. list_ids()          -- get record IDs                    |
|    7. clear()             -- bulk delete                       |
|    8. stats()             -- database statistics               |
|    9. close()             -- shut down + context manager       |
|                                                                |
|  Run with:                                                     |
|    pytest tests/test_day7.py -v                                |
|    python -m pytest tests/test_day7.py -v                      |
+===============================================================+

WHAT THESE TESTS VERIFY:
    - delete() removes record from both SQLite and vector matrix
    - delete() non-existent ID returns False
    - update() with new text re-embeds and replaces vector
    - update() with new metadata replaces metadata in SQLite
    - create_collection creates and list_collections shows it
    - delete_collection removes collection and all its records
    - Cannot delete "default" collection
    - clear() removes all records, count() returns 0
    - stats() returns correct numbers
    - Context manager (with statement) works
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------
# Make sure Python can find our project modules.
# ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ARCHITECTURE import VectorRecord, CollectionInfo, DatabaseStats
from core.embeddings import SimpleEmbeddingEngine
from core.vector_store import VectorStore


# ===============================================================
# FIXTURES
# ===============================================================

@pytest.fixture
def tmp_store(tmp_path):
    """
    Create a temporary VectorStore pre-loaded with 3 test records.

    Uses SimpleEmbeddingEngine so tests are fast and deterministic.
    """
    storage_dir = str(tmp_path / "crud_test")
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    # Force fallback engine for deterministic, lightweight tests.
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)

    # Insert three records with known IDs and metadata.
    store.insert(
        "machine learning algorithms process data",
        metadata={"category": "tech", "year": "2024"},
        id="rec_001",
    )
    store.insert(
        "the cat sat on the warm mat",
        metadata={"category": "animals"},
        id="rec_002",
    )
    store.insert(
        "quantum physics explores particles",
        metadata={"category": "science", "year": "2024"},
        id="rec_003",
    )
    yield store
    store.db.close()


@pytest.fixture
def empty_store(tmp_path):
    """Create an empty VectorStore for edge-case tests."""
    storage_dir = str(tmp_path / "empty_test")
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
    yield store
    store.db.close()


# ===============================================================
# DELETE TESTS
# ===============================================================

class TestDelete:
    """Tests for VectorStore.delete()."""

    def test_delete_removes_from_sqlite_and_vectors(self, tmp_store):
        """Deleting a record removes it from both SQLite and NumPy."""
        # Verify the record exists before deletion.
        assert tmp_store.get("rec_001") is not None
        assert tmp_store.count() == 3
        original_shape = tmp_store._vectors.shape[0]

        # Delete it.
        result = tmp_store.delete("rec_001")

        assert result is True
        # Gone from SQLite (get returns None).
        assert tmp_store.get("rec_001") is None
        # Count decreased by 1.
        assert tmp_store.count() == 2
        # Vector matrix lost one row.
        assert tmp_store._vectors.shape[0] == original_shape - 1
        # ID no longer in the mapping.
        assert "rec_001" not in tmp_store._id_list
        assert "rec_001" not in tmp_store._id_to_index

    def test_delete_nonexistent_returns_false(self, tmp_store):
        """Deleting an ID that doesn't exist returns False, no error."""
        result = tmp_store.delete("rec_nonexistent")
        assert result is False
        # Nothing else changed.
        assert tmp_store.count() == 3

    def test_delete_cascades_metadata(self, tmp_store):
        """Deleting a record also removes its metadata (cascade)."""
        # rec_001 has metadata {"category": "tech", "year": "2024"}.
        meta_before = tmp_store.db.get_metadata("rec_001")
        assert len(meta_before) == 2

        tmp_store.delete("rec_001")

        # Metadata for that ID should be gone.
        meta_after = tmp_store.db.get_metadata("rec_001")
        assert len(meta_after) == 0

    def test_remaining_records_still_work_after_delete(self, tmp_store):
        """After deleting rec_001, rec_002 and rec_003 are still accessible."""
        tmp_store.delete("rec_001")

        rec2 = tmp_store.get("rec_002")
        assert rec2 is not None
        assert rec2.text == "the cat sat on the warm mat"

        rec3 = tmp_store.get("rec_003")
        assert rec3 is not None
        assert rec3.text == "quantum physics explores particles"


# ===============================================================
# UPDATE TESTS
# ===============================================================

class TestUpdate:
    """Tests for VectorStore.update()."""

    def test_update_text_reembeds_vector(self, tmp_store):
        """Updating text re-embeds and replaces the vector."""
        old_record = tmp_store.get("rec_001")
        old_vector = old_record.vector.copy()

        result = tmp_store.update("rec_001", text="completely new text here")

        assert result is True
        new_record = tmp_store.get("rec_001")
        # Text should be updated.
        assert new_record.text == "completely new text here"
        # Vector should be different (different text → different embedding).
        assert not np.array_equal(old_vector, new_record.vector)

    def test_update_metadata_replaces_in_sqlite(self, tmp_store):
        """Updating metadata does a full replace (not merge)."""
        # Original metadata: {"category": "tech", "year": "2024"}
        result = tmp_store.update(
            "rec_001",
            metadata={"new_key": "new_value"},
        )

        assert result is True
        record = tmp_store.get("rec_001")
        # Old keys are gone, only new keys remain.
        assert record.metadata == {"new_key": "new_value"}

    def test_update_both_text_and_metadata(self, tmp_store):
        """Can update text and metadata in one call."""
        result = tmp_store.update(
            "rec_002",
            text="updated cat text",
            metadata={"status": "edited"},
        )

        assert result is True
        record = tmp_store.get("rec_002")
        assert record.text == "updated cat text"
        assert record.metadata == {"status": "edited"}

    def test_update_nonexistent_returns_false(self, tmp_store):
        """Updating a non-existent ID returns False."""
        result = tmp_store.update("rec_nonexistent", text="whatever")
        assert result is False

    def test_update_metadata_to_empty(self, tmp_store):
        """Passing metadata={} clears all metadata."""
        tmp_store.update("rec_001", metadata={})
        record = tmp_store.get("rec_001")
        assert record.metadata == {}


# ===============================================================
# COLLECTION MANAGEMENT TESTS
# ===============================================================

class TestCollections:
    """Tests for create_collection, list_collections, delete_collection."""

    def test_create_collection(self, empty_store):
        """Creating a new collection returns CollectionInfo."""
        info = empty_store.create_collection("science", "Science papers")

        assert isinstance(info, CollectionInfo)
        assert info.name == "science"
        assert info.description == "Science papers"
        assert info.count == 0
        assert info.dimension == 384

    def test_create_duplicate_raises(self, empty_store):
        """Creating a collection that already exists raises ValueError."""
        empty_store.create_collection("papers")

        with pytest.raises(ValueError, match="already exists"):
            empty_store.create_collection("papers")

    def test_list_collections_shows_new_collection(self, empty_store):
        """list_collections includes newly created collections."""
        empty_store.create_collection("research")

        collections = empty_store.list_collections()
        names = [c.name for c in collections]

        # "default" always exists + our new one.
        assert "default" in names
        assert "research" in names

    def test_list_collections_shows_record_counts(self, tmp_store):
        """list_collections shows correct record count per collection."""
        collections = tmp_store.list_collections()
        default_col = [c for c in collections if c.name == "default"][0]

        # tmp_store has 3 records in "default".
        assert default_col.count == 3

    def test_delete_collection_removes_records(self, tmp_store):
        """Deleting a collection removes it and all its records."""
        # Create a new collection and insert a record into it.
        tmp_store.create_collection("temp_col")
        tmp_store.insert(
            "temporary data",
            id="temp_001",
            collection="temp_col",
        )
        assert tmp_store.count() == 4

        # Delete the collection.
        result = tmp_store.delete_collection("temp_col")
        assert result is True

        # Record in that collection is gone.
        assert tmp_store.get("temp_001") is None
        assert tmp_store.count() == 3

        # Collection no longer listed.
        names = [c.name for c in tmp_store.list_collections()]
        assert "temp_col" not in names

    def test_cannot_delete_default_collection(self, tmp_store):
        """Deleting the 'default' collection raises ValueError."""
        with pytest.raises(ValueError, match="Cannot delete"):
            tmp_store.delete_collection("default")

    def test_delete_nonexistent_collection_returns_false(self, tmp_store):
        """Deleting a collection that doesn't exist returns False."""
        result = tmp_store.delete_collection("no_such_collection")
        assert result is False


# ===============================================================
# LIST IDS TESTS
# ===============================================================

class TestListIds:
    """Tests for VectorStore.list_ids()."""

    def test_list_all_ids(self, tmp_store):
        """list_ids() with no args returns all IDs."""
        ids = tmp_store.list_ids()
        assert set(ids) == {"rec_001", "rec_002", "rec_003"}

    def test_list_ids_with_limit(self, tmp_store):
        """list_ids(limit=2) returns at most 2 IDs."""
        ids = tmp_store.list_ids(limit=2)
        assert len(ids) == 2

    def test_list_ids_empty_db(self, empty_store):
        """list_ids() on empty database returns empty list."""
        ids = empty_store.list_ids()
        assert ids == []


# ===============================================================
# CLEAR TESTS
# ===============================================================

class TestClear:
    """Tests for VectorStore.clear()."""

    def test_clear_all(self, tmp_store):
        """clear() with no args removes all records."""
        count = tmp_store.clear()

        assert count == 3
        assert tmp_store.count() == 0
        assert tmp_store._vectors.shape[0] == 0
        assert tmp_store._id_list == []

    def test_clear_specific_collection(self, tmp_store):
        """clear(collection) removes only records in that collection."""
        # Create another collection and insert into it.
        tmp_store.create_collection("other")
        tmp_store.insert("other data", id="other_001", collection="other")
        assert tmp_store.count() == 4

        # Clear only "default".
        count = tmp_store.clear(collection="default")

        assert count == 3
        assert tmp_store.count() == 1
        # The record in "other" survives.
        assert tmp_store.get("other_001") is not None

    def test_clear_empty_database(self, empty_store):
        """clear() on empty database returns 0."""
        count = empty_store.clear()
        assert count == 0


# ===============================================================
# STATS TESTS
# ===============================================================

class TestStats:
    """Tests for VectorStore.stats()."""

    def test_stats_returns_correct_numbers(self, tmp_store):
        """stats() returns a DatabaseStats with correct values."""
        s = tmp_store.stats()

        assert isinstance(s, DatabaseStats)
        assert s.total_records == 3
        assert s.total_collections == 1  # just "default"
        assert s.dimension == 384
        # Memory: 3 records * 384 dims * 4 bytes/float32 = 4608
        assert s.memory_usage_bytes == 3 * 384 * 4
        assert s.storage_path == tmp_store.storage_path
        assert "minivecdb.db" in s.db_file

    def test_stats_empty_database(self, empty_store):
        """stats() on empty database shows zeros."""
        s = empty_store.stats()
        assert s.total_records == 0
        assert s.memory_usage_bytes == 0


# ===============================================================
# CONTEXT MANAGER TESTS
# ===============================================================

class TestContextManager:
    """Tests for the `with VectorStore(...) as db:` pattern."""

    def test_context_manager_works(self, tmp_path):
        """with statement opens and closes the store."""
        storage_dir = str(tmp_path / "ctx_test")

        with VectorStore(storage_path=storage_dir) as store:
            store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
            store.insert("context manager test", id="ctx_001")
            assert store.count() == 1

        # After the with block, the store should be closed.
        # Verify data was saved by reopening.
        store2 = VectorStore(storage_path=storage_dir)
        assert store2.count() == 1
        store2.db.close()

    def test_enter_returns_self(self, tmp_path):
        """__enter__ returns the VectorStore instance."""
        storage_dir = str(tmp_path / "enter_test")
        store = VectorStore(storage_path=storage_dir)

        result = store.__enter__()
        assert result is store

        store.close()
