"""End-to-end integration tests for the complete MiniVecDB pipeline.

This module verifies the full system flow using a temporary storage directory:
collection management, insert/search/update/delete, persistence, stats, and clear.
The same flow can run under pytest and as a standalone script.
"""

import os
import shutil
import sys
import tempfile
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.distance_metrics import cosine_similarity
from core.embeddings import SimpleEmbeddingEngine
from core.vector_store import VectorStore


def _make_store(storage_dir: str) -> VectorStore:
    """Create a VectorStore that uses a deterministic lightweight embedding engine."""
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
    return store


def _assert_ranked_results(results: List, metric_name: str, descending: bool) -> None:
    """Validate search results are non-empty, finite, ranked, and metric-consistent."""
    assert len(results) > 0, f"Expected non-empty results for metric={metric_name}."

    scores = [r.score for r in results]
    ranks = [r.rank for r in results]
    metrics = [r.metric for r in results]

    assert all(np.isfinite(score) for score in scores)
    assert ranks == list(range(1, len(results) + 1))
    assert all(metric == metric_name for metric in metrics)

    for idx in range(len(scores) - 1):
        if descending:
            assert scores[idx] >= scores[idx + 1]
        else:
            assert scores[idx] <= scores[idx + 1]


def run_full_pipeline_integration(verbose: bool = False) -> None:
    """Run the complete integration scenario requested for MiniVecDB.

    Steps covered:
    1) create store in temp directory
    2) create collections
    3) insert records into each collection
    4) validate counts
    5) run searches/filters/metrics
    6) get/update/delete records and collections
    7) save/close/reopen and re-validate
    8) validate stats
    9) clear all records
    """
    temp_dir = tempfile.mkdtemp(prefix="minivecdb_integration_")
    store = None

    science_docs = [
        "Quantum physics research explores particles and fields.",
        "Physics lab results improve quantum research methods.",
        "Scientific research in astrophysics studies galaxies.",
        "Biology and chemistry research drive science discovery.",
        "Modern science uses data analysis for research breakthroughs.",
    ]
    sports_docs = [
        "Football teams train hard before the championship season.",
        "Basketball players practice defense and shooting drills.",
        "Olympic athletes improve speed, stamina, and strength.",
        "Tennis tournaments reward precision, timing, and fitness.",
        "Cricket strategy depends on bowling rhythm and teamwork.",
    ]

    science_ids = [f"science_{i}" for i in range(1, 6)]
    sports_ids = [f"sports_{i}" for i in range(1, 6)]

    try:
        if verbose:
            print("[1] Creating VectorStore in temporary directory")
        store = _make_store(temp_dir)

        if verbose:
            print("[2] Creating collections: science, sports")
        store.create_collection("science", "Science collection")
        store.create_collection("sports", "Sports collection")

        if verbose:
            print("[3] Inserting 5 science records")
        for record_id, text in zip(science_ids, science_docs):
            inserted_id = store.insert(
                text=text,
                metadata={"category": "science", "year": "2024"},
                id=record_id,
                collection="science",
            )
            assert inserted_id == record_id

        if verbose:
            print("[4] Inserting 5 sports records")
        for record_id, text in zip(sports_ids, sports_docs):
            inserted_id = store.insert(
                text=text,
                metadata={"category": "sports", "year": "2023"},
                id=record_id,
                collection="sports",
            )
            assert inserted_id == record_id

        if verbose:
            print("[5] Verifying counts (10 total, 5 per collection)")
        assert store.count() == 10
        assert store.count("science") == 5
        assert store.count("sports") == 5

        if verbose:
            print("[6] Querying semantic search: quantum physics research")
        query = "quantum physics research"
        results = store.search(query, top_k=5)
        assert len(results) > 0
        assert results[0].record.collection == "science"

        if verbose:
            print("[7] Searching with metadata filter category=science")
        filtered_results = store.search(query, top_k=5, filters={"category": "science"})
        assert len(filtered_results) > 0
        assert all(r.record.collection == "science" for r in filtered_results)
        assert all(r.record.metadata.get("category") == "science" for r in filtered_results)

        if verbose:
            print("[8] Searching with metric=euclidean")
        euclidean_results = store.search(query, top_k=5, metric="euclidean")
        _assert_ranked_results(euclidean_results, metric_name="euclidean", descending=False)

        if verbose:
            print("[9] Searching with metric=dot")
        dot_results = store.search(query, top_k=5, metric="dot")
        _assert_ranked_results(dot_results, metric_name="dot", descending=True)

        target_id = science_ids[0]
        if verbose:
            print("[10] Getting a specific record by ID")
        target_record = store.get(target_id)
        assert target_record is not None
        assert target_record.id == target_id
        assert target_record.text == science_docs[0]
        assert target_record.collection == "science"
        assert target_record.metadata == {"category": "science", "year": "2024"}
        old_vector = target_record.vector.copy()

        if verbose:
            print("[11] Updating record text and verifying vector changed")
        updated_text = "Marine biology investigates coral reef ecosystems."
        assert store.update(target_id, text=updated_text) is True
        updated_record = store.get(target_id)
        assert updated_record is not None
        assert updated_record.text == updated_text
        cosine_after_update = cosine_similarity(old_vector, updated_record.vector)
        assert cosine_after_update < 1.0

        if verbose:
            print("[12] Updating metadata and verifying SQLite values")
        updated_metadata = {"category": "science", "year": "2025", "reviewed": "yes"}
        assert store.update(target_id, metadata=updated_metadata) is True
        sqlite_metadata = store.db.get_metadata(target_id)
        assert sqlite_metadata == updated_metadata

        if verbose:
            print("[13] Deleting one sports record")
        deleted_record_id = sports_ids[0]
        assert store.delete(deleted_record_id) is True
        assert store.count() == 9
        assert store.get(deleted_record_id) is None

        if verbose:
            print("[14] Deleting sports collection and verifying cascade behavior")
        assert store.delete_collection("sports") is True
        assert store.db.collection_exists("sports") is False
        assert store.count("sports") == 0
        for record_id in sports_ids[1:]:
            assert store.get(record_id) is None

        if verbose:
            print("[15] Saving, closing, and reopening")
        store.save()
        store.close()
        store = _make_store(temp_dir)
        assert store.count() == 5
        reopened_results = store.search("science research", top_k=3)
        assert len(reopened_results) > 0
        assert reopened_results[0].record.collection == "science"

        if verbose:
            print("[16] Verifying stats() numbers")
        stats = store.stats()
        assert stats.total_records == 5
        assert stats.total_collections == 2  # default + science
        assert stats.dimension == 384
        assert stats.memory_usage_bytes == 5 * 384 * 4

        if verbose:
            print("[17] Clearing remaining records")
        cleared = store.clear()
        assert cleared == 5
        assert store.count() == 0
        assert store._vectors.shape == (0, 384)

    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_full_pipeline_integration() -> None:
    """Pytest entry point for the full integration pipeline test."""
    run_full_pipeline_integration(verbose=False)


if __name__ == "__main__":
    """Run the full integration test as a standalone script."""
    try:
        run_full_pipeline_integration(verbose=True)
        print("=" * 72)
        print("INTEGRATION SUMMARY: 1 passed, 0 failed")
        print("=" * 72)
    except AssertionError as exc:
        print("=" * 72)
        print(f"INTEGRATION SUMMARY: 0 passed, 1 failed ({exc})")
        print("=" * 72)
        raise SystemExit(1)
