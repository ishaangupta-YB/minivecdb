"""
+===============================================================+
|  MiniVecDB -- Day 8 Tests                                      |
|  File: minivecdb/tests/test_day8.py                            |
|                                                                |
|  Tests for PERSISTENCE -- the save/load system that makes the  |
|  database survive restarts:                                    |
|    1. save() + _load_vectors()  -- round-trip persistence      |
|    2. _rebuild_vectors()        -- emergency rebuild from SQL   |
|    3. Multiple open/close cycles                               |
|                                                                |
|  Every test uses tempfile.mkdtemp() for isolated storage.      |
|                                                                |
|  Run with:                                                     |
|    pytest tests/test_day8.py -v                                |
|    python -m pytest tests/test_day8.py -v                      |
+===============================================================+

WHAT THESE TESTS VERIFY:
    - Insert records -> close -> reopen -> records still there
    - Insert records -> close -> reopen -> search still works
    - Insert records -> close -> reopen -> count() matches
    - Delete a record -> close -> reopen -> deleted record stays deleted
    - Metadata survives restart (check via get())
    - Empty database saves/loads cleanly
    - Corrupted vectors.npy triggers rebuild (delete the .npy file)
    - Multiple open/close cycles work without data loss
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pytest

# ---------------------------------------------------------------
# Make sure Python can find our project modules.
# ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ARCHITECTURE import VectorRecord
from core.embeddings import SimpleEmbeddingEngine
from core.vector_store import VectorStore


# ===============================================================
# HELPER -- create a VectorStore with SimpleEmbeddingEngine
# ===============================================================

def make_store(storage_dir: str) -> VectorStore:
    """
    Create a VectorStore at the given path using the lightweight
    SimpleEmbeddingEngine so tests are fast and deterministic.
    """
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
    return store


# ===============================================================
# FIXTURES
# ===============================================================

@pytest.fixture
def storage_dir():
    """
    Create a unique temporary directory for each test.

    Uses tempfile.mkdtemp() instead of pytest's tmp_path to match
    the spec requirement.  The directory is cleaned up after the test.
    """
    d = tempfile.mkdtemp(prefix="minivecdb_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ===============================================================
# BASIC PERSISTENCE TESTS -- Data survives close/reopen
# ===============================================================

class TestBasicPersistence:
    """Verify that data persists across close() and reopen."""

    def test_records_survive_restart(self, storage_dir):
        """Insert records -> close -> reopen -> records still there."""
        # --- Session 1: insert data ---
        store = make_store(storage_dir)
        store.insert("hello world", id="rec_001", metadata={"lang": "en"})
        store.insert("goodbye world", id="rec_002", metadata={"lang": "en"})
        store.close()

        # --- Session 2: reopen and verify ---
        store2 = make_store(storage_dir)
        assert store2.count() == 2
        assert store2.get("rec_001") is not None
        assert store2.get("rec_002") is not None
        store2.close()

    def test_count_matches_after_restart(self, storage_dir):
        """count() returns the same number after restart."""
        store = make_store(storage_dir)
        store.insert("text one", id="a1")
        store.insert("text two", id="a2")
        store.insert("text three", id="a3")
        count_before = store.count()
        store.close()

        store2 = make_store(storage_dir)
        assert store2.count() == count_before
        assert store2.count() == 3
        store2.close()

    def test_search_works_after_restart(self, storage_dir):
        """Insert records -> close -> reopen -> search returns results."""
        store = make_store(storage_dir)
        store.insert("machine learning algorithms", id="ml_001")
        store.insert("deep learning neural networks", id="ml_002")
        store.insert("cat sat on the mat", id="cat_001")
        store.close()

        store2 = make_store(storage_dir)
        results = store2.search("machine learning", top_k=2)
        assert len(results) > 0
        # The top result should be one of the ML texts (they share
        # more words with the query than the cat text).
        top_ids = [r.record.id for r in results]
        assert "ml_001" in top_ids or "ml_002" in top_ids
        store2.close()

    def test_metadata_survives_restart(self, storage_dir):
        """Metadata key-value pairs persist across restarts."""
        store = make_store(storage_dir)
        store.insert(
            "tagged record",
            id="meta_001",
            metadata={"category": "science", "year": "2024"},
        )
        store.close()

        store2 = make_store(storage_dir)
        record = store2.get("meta_001")
        assert record is not None
        assert record.metadata == {"category": "science", "year": "2024"}
        store2.close()

    def test_delete_survives_restart(self, storage_dir):
        """Delete a record -> close -> reopen -> deleted record stays gone."""
        store = make_store(storage_dir)
        store.insert("keep me", id="keep_001")
        store.insert("delete me", id="del_001")
        store.delete("del_001")
        store.close()

        store2 = make_store(storage_dir)
        assert store2.count() == 1
        assert store2.get("keep_001") is not None
        assert store2.get("del_001") is None
        store2.close()


# ===============================================================
# EMPTY DATABASE PERSISTENCE
# ===============================================================

class TestEmptyPersistence:
    """Verify that an empty database saves and loads cleanly."""

    def test_empty_save_load(self, storage_dir):
        """Empty database -> close -> reopen -> still empty, no errors."""
        store = make_store(storage_dir)
        assert store.count() == 0
        store.save()
        store.close()

        store2 = make_store(storage_dir)
        assert store2.count() == 0
        assert store2._vectors.shape == (0, 384)
        assert store2._id_list == []
        store2.close()

    def test_insert_then_clear_then_restart(self, storage_dir):
        """Insert -> clear -> close -> reopen -> still empty."""
        store = make_store(storage_dir)
        store.insert("temporary", id="tmp_001")
        store.clear()
        store.close()

        store2 = make_store(storage_dir)
        assert store2.count() == 0
        assert store2.get("tmp_001") is None
        store2.close()


# ===============================================================
# REBUILD TESTS -- Corrupted or missing .npy files
# ===============================================================

class TestRebuild:
    """Verify _rebuild_vectors() recovers from corrupted state."""

    def test_missing_npy_triggers_rebuild(self, storage_dir):
        """
        Deleting vectors.npy triggers a rebuild from SQLite.

        Steps:
          1. Insert records and close (saves .npy + .json).
          2. Delete the vectors.npy file to simulate corruption.
          3. Reopen -- _load_vectors detects the missing file and
             calls _rebuild_vectors() which re-embeds all texts.
          4. Verify all records are still accessible.
        """
        store = make_store(storage_dir)
        store.insert("rebuild test alpha", id="rb_001")
        store.insert("rebuild test beta", id="rb_002")
        store.close()

        # --- Corrupt the state: delete vectors.npy ---
        npy_path = os.path.join(storage_dir, "vectors.npy")
        assert os.path.exists(npy_path)
        os.remove(npy_path)
        assert not os.path.exists(npy_path)

        # --- Reopen: should rebuild automatically ---
        store2 = make_store(storage_dir)
        assert store2.count() == 2
        assert store2.get("rb_001") is not None
        assert store2.get("rb_002") is not None
        # Vectors should be rebuilt (shape matches record count).
        assert store2._vectors.shape[0] == 2
        store2.close()

    def test_missing_id_mapping_triggers_rebuild(self, storage_dir):
        """Deleting id_mapping.json also triggers rebuild."""
        store = make_store(storage_dir)
        store.insert("mapping test", id="map_001")
        store.close()

        json_path = os.path.join(storage_dir, "id_mapping.json")
        os.remove(json_path)

        store2 = make_store(storage_dir)
        assert store2.count() == 1
        assert store2.get("map_001") is not None
        store2.close()

    def test_both_files_missing_triggers_rebuild(self, storage_dir):
        """Deleting both .npy and .json triggers rebuild."""
        store = make_store(storage_dir)
        store.insert("both missing test", id="both_001")
        store.close()

        os.remove(os.path.join(storage_dir, "vectors.npy"))
        os.remove(os.path.join(storage_dir, "id_mapping.json"))

        store2 = make_store(storage_dir)
        assert store2.count() == 1
        assert store2.get("both_001") is not None
        store2.close()

    def test_rebuild_preserves_search(self, storage_dir):
        """After rebuild, search still returns meaningful results."""
        store = make_store(storage_dir)
        store.insert("machine learning data science", id="ml_r1")
        store.insert("cat sat on the mat", id="cat_r1")
        store.close()

        # Delete .npy to force rebuild.
        os.remove(os.path.join(storage_dir, "vectors.npy"))

        store2 = make_store(storage_dir)
        results = store2.search("machine learning", top_k=2)
        assert len(results) > 0
        store2.close()


# ===============================================================
# MULTIPLE OPEN/CLOSE CYCLES
# ===============================================================

class TestMultipleCycles:
    """Verify that repeated open/close cycles don't corrupt data."""

    def test_three_cycles_no_data_loss(self, storage_dir):
        """Open/close three times with incremental inserts."""
        # Cycle 1: insert 2 records.
        store = make_store(storage_dir)
        store.insert("first cycle alpha", id="c1_a")
        store.insert("first cycle beta", id="c1_b")
        store.close()

        # Cycle 2: insert 1 more record.
        store = make_store(storage_dir)
        assert store.count() == 2
        store.insert("second cycle gamma", id="c2_g")
        store.close()

        # Cycle 3: verify all 3 records survive.
        store = make_store(storage_dir)
        assert store.count() == 3
        assert store.get("c1_a") is not None
        assert store.get("c1_b") is not None
        assert store.get("c2_g") is not None
        store.close()

    def test_delete_across_cycles(self, storage_dir):
        """Insert in cycle 1, delete in cycle 2, verify in cycle 3."""
        # Cycle 1: insert.
        store = make_store(storage_dir)
        store.insert("to delete", id="del_x")
        store.insert("to keep", id="keep_x")
        store.close()

        # Cycle 2: delete one.
        store = make_store(storage_dir)
        store.delete("del_x")
        assert store.count() == 1
        store.close()

        # Cycle 3: verify.
        store = make_store(storage_dir)
        assert store.count() == 1
        assert store.get("del_x") is None
        assert store.get("keep_x") is not None
        store.close()

    def test_update_across_cycles(self, storage_dir):
        """Update in one cycle, verify the change in the next."""
        store = make_store(storage_dir)
        store.insert("original text", id="upd_001", metadata={"v": "1"})
        store.close()

        store = make_store(storage_dir)
        store.update("upd_001", text="updated text", metadata={"v": "2"})
        store.close()

        store = make_store(storage_dir)
        record = store.get("upd_001")
        assert record.text == "updated text"
        assert record.metadata == {"v": "2"}
        store.close()

    def test_context_manager_persists(self, storage_dir):
        """Using `with` block saves data that survives reopening."""
        with make_store(storage_dir) as store:
            store.insert("context data", id="ctx_001")

        store2 = make_store(storage_dir)
        assert store2.count() == 1
        assert store2.get("ctx_001").text == "context data"
        store2.close()


class TestManagedRunDefaults:
    """Verify managed db_run defaults preserve persistence expectations."""

    def test_default_run_reused_across_reopen(self, tmp_path, monkeypatch):
        """VectorStore() should reopen the same active run by default."""
        monkeypatch.setenv("MINIVECDB_PROJECT_ROOT", str(tmp_path))

        with VectorStore() as store:
            store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
            run_path = store.storage_path
            store.insert("managed persistence", id="managed_001")

        with VectorStore() as reopened:
            assert reopened.storage_path == run_path
            record = reopened.get("managed_001")
            assert record is not None
            assert record.text == "managed persistence"

    def test_new_run_creates_isolated_dataset(self, tmp_path, monkeypatch):
        """new_run=True should switch to a clean run directory."""
        monkeypatch.setenv("MINIVECDB_PROJECT_ROOT", str(tmp_path))

        with VectorStore() as store:
            store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
            first_run = store.storage_path
            store.insert("first run record", id="run_a")

        with VectorStore(new_run=True) as second_store:
            second_store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
            second_run = second_store.storage_path
            assert second_run != first_run
            assert second_store.get("run_a") is None
