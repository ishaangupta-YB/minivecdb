"""
+===============================================================+
|  MiniVecDB -- Day 6 Tests                                      |
|  File: minivecdb/tests/test_day6.py                            |
|                                                                |
|  Tests for the SEARCH ENGINE -- the core feature that makes    |
|  MiniVecDB a vector database:                                  |
|    1. search()           -- text query -> ranked results       |
|    2. search_by_vector() -- vector query -> ranked results     |
|    3. _get_filtered_indices() -- metadata pre-filtering        |
|    4. __len__()          -- len(store) support                 |
|                                                                |
|  Run with:                                                     |
|    pytest tests/test_day6.py -v                                |
|    python -m pytest tests/test_day6.py -v                      |
+===============================================================+

WHAT THESE TESTS VERIFY:
    - search() returns results sorted by relevance
    - search() with top_k=3 returns exactly 3 results
    - search() result scores are between -1 and 1 for cosine
    - search() with metric="euclidean" returns results sorted ascending
    - search() with metric="dot" works correctly
    - search_by_vector() returns same results as search() for same query
    - search() on empty database raises ValueError
    - SearchResult objects have correct rank (1, 2, 3, ...)
    - Each result's .record has valid text and metadata
    - search() with filters returns only matching records
    - __len__ returns the correct count
"""

import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------
# Make sure Python can find our project modules.
# ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ARCHITECTURE import VectorRecord, SearchResult
from core.embeddings import SimpleEmbeddingEngine
from core.vector_store import VectorStore


# ===============================================================
# FIXTURES
# ===============================================================

# A fixed set of texts with clear semantic groupings so we can
# predict which results should rank highest.  The SimpleEmbeddingEngine
# (bag-of-words fallback) matches on word overlap, so we use texts
# that share specific words to create predictable similarity.

SCIENCE_TEXTS = [
    "machine learning algorithms process large datasets",
    "deep learning neural networks train on data",
    "the cat sat quietly on the warm mat",
    "dogs love to play fetch in the park",
    "quantum physics explores subatomic particles",
]

SCIENCE_METADATA = [
    {"category": "tech", "topic": "ml"},
    {"category": "tech", "topic": "dl"},
    {"category": "animals", "topic": "cats"},
    {"category": "animals", "topic": "dogs"},
    {"category": "science", "topic": "physics"},
]


@pytest.fixture
def tmp_store(tmp_path):
    """
    Create a temporary VectorStore pre-loaded with test data.

    Uses SimpleEmbeddingEngine (bag-of-words fallback) so tests
    don't require the 80 MB sentence-transformers model download.
    """
    storage_dir = str(tmp_path / "search_test")
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    # Force fallback engine for deterministic, lightweight tests.
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
    # Insert all test records with metadata
    store.insert_batch(
        texts=SCIENCE_TEXTS,
        metadata_list=SCIENCE_METADATA,
    )
    yield store
    store.db.close()


@pytest.fixture
def empty_store(tmp_path):
    """Create an empty VectorStore for testing edge cases."""
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
# TESTS: search() -- Basic behavior
# ===============================================================

class TestSearchBasic:
    """Test that search() returns correctly structured results."""

    def test_search_returns_results(self, tmp_store):
        """search() should return a non-empty list of SearchResult."""
        results = tmp_store.search("machine learning")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_top_k_limits_results(self, tmp_store):
        """search() with top_k=3 should return exactly 3 results."""
        results = tmp_store.search("machine learning", top_k=3)
        assert len(results) == 3

    def test_search_top_k_larger_than_db(self, tmp_store):
        """top_k larger than the DB should return all records."""
        results = tmp_store.search("test query", top_k=100)
        assert len(results) == len(SCIENCE_TEXTS)

    def test_search_default_top_k_is_5(self, tmp_store):
        """Default top_k=5 should return 5 results when DB has 5 records."""
        results = tmp_store.search("some query")
        assert len(results) == 5


# ===============================================================
# TESTS: search() -- Sorting and relevance
# ===============================================================

class TestSearchSorting:
    """Test that results are sorted by the chosen metric."""

    def test_cosine_scores_sorted_descending(self, tmp_store):
        """Cosine results should be sorted highest-score-first."""
        results = tmp_store.search("learning algorithms data", metric="cosine")
        scores = [r.score for r in results]
        # Each score should be >= the next one (descending order)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Cosine scores not sorted descending: {scores}"
            )

    def test_euclidean_scores_sorted_ascending(self, tmp_store):
        """Euclidean results should be sorted lowest-distance-first."""
        results = tmp_store.search("learning data", metric="euclidean")
        scores = [r.score for r in results]
        # Each score should be <= the next one (ascending order)
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Euclidean scores not sorted ascending: {scores}"
            )

    def test_dot_scores_sorted_descending(self, tmp_store):
        """Dot product results should be sorted highest-first."""
        results = tmp_store.search("learning neural networks", metric="dot")
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Dot product scores not sorted descending: {scores}"
            )


# ===============================================================
# TESTS: search() -- Score ranges
# ===============================================================

class TestSearchScores:
    """Test that scores are in valid ranges for each metric."""

    def test_cosine_scores_in_valid_range(self, tmp_store):
        """Cosine similarity scores should be between -1 and 1."""
        results = tmp_store.search("test query", metric="cosine")
        for r in results:
            assert -1.0 <= r.score <= 1.0, (
                f"Cosine score {r.score} out of range [-1, 1]"
            )

    def test_euclidean_scores_non_negative(self, tmp_store):
        """Euclidean distance scores should be >= 0."""
        results = tmp_store.search("test query", metric="euclidean")
        for r in results:
            assert r.score >= 0.0, (
                f"Euclidean score {r.score} should be non-negative"
            )

    def test_dot_metric_works(self, tmp_store):
        """Dot product search should return valid results."""
        results = tmp_store.search("learning data", metric="dot")
        assert len(results) > 0
        # Dot product scores are unbounded but should be finite
        for r in results:
            assert np.isfinite(r.score), (
                f"Dot product score {r.score} is not finite"
            )


# ===============================================================
# TESTS: search() -- SearchResult structure
# ===============================================================

class TestSearchResultStructure:
    """Test that SearchResult objects are correctly populated."""

    def test_ranks_are_sequential(self, tmp_store):
        """Ranks should be 1, 2, 3, ... (1-based, sequential)."""
        results = tmp_store.search("learning algorithms", top_k=5)
        expected_ranks = list(range(1, len(results) + 1))
        actual_ranks = [r.rank for r in results]
        assert actual_ranks == expected_ranks

    def test_metric_field_matches_request(self, tmp_store):
        """Each result's .metric should match the requested metric."""
        for metric_name in ["cosine", "euclidean", "dot"]:
            results = tmp_store.search("test", metric=metric_name)
            for r in results:
                assert r.metric == metric_name

    def test_record_has_valid_text(self, tmp_store):
        """Each result's .record should have non-empty text."""
        results = tmp_store.search("machine learning")
        for r in results:
            assert isinstance(r.record, VectorRecord)
            assert isinstance(r.record.text, str)
            assert len(r.record.text) > 0

    def test_record_has_valid_metadata(self, tmp_store):
        """Each result's .record should have a metadata dict."""
        results = tmp_store.search("machine learning")
        for r in results:
            assert isinstance(r.record.metadata, dict)
            # All our test records have metadata
            assert len(r.record.metadata) > 0

    def test_record_has_vector(self, tmp_store):
        """Each result's .record should carry its 384-dim vector."""
        results = tmp_store.search("test", top_k=2)
        for r in results:
            assert r.record.vector.shape == (384,)
            assert r.record.vector.dtype == np.float32


# ===============================================================
# TESTS: search() -- Empty database
# ===============================================================

class TestSearchEmpty:
    """Test search behavior on an empty database."""

    def test_search_empty_raises_valueerror(self, empty_store):
        """search() on an empty database should raise ValueError."""
        with pytest.raises(ValueError, match="empty database"):
            empty_store.search("anything")

    def test_search_by_vector_empty_raises_valueerror(self, empty_store):
        """search_by_vector() on empty DB should raise ValueError."""
        dummy_vector = np.random.rand(384).astype(np.float32)
        with pytest.raises(ValueError, match="empty database"):
            empty_store.search_by_vector(dummy_vector)


# ===============================================================
# TESTS: search_by_vector()
# ===============================================================

class TestSearchByVector:
    """Test searching with a pre-computed vector."""

    def test_search_by_vector_returns_results(self, tmp_store):
        """search_by_vector() should return SearchResult objects."""
        # Use one of the stored vectors as the query
        query_vec = tmp_store._vectors[0].copy()

        results = tmp_store.search_by_vector(query_vec, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_by_vector_matches_search(self, tmp_store):
        """search_by_vector() should return the same ranking as search()
        when given the same embedded vector."""
        query_text = "machine learning algorithms"
        # Get the vector for this query
        query_vec = tmp_store.embedding_engine.encode(query_text)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        # Search both ways
        text_results = tmp_store.search(query_text, top_k=5)
        vec_results = tmp_store.search_by_vector(query_vec, top_k=5)

        # The record IDs should be in the same order
        text_ids = [r.record.id for r in text_results]
        vec_ids = [r.record.id for r in vec_results]
        assert text_ids == vec_ids

        # Scores should be nearly identical
        for tr, vr in zip(text_results, vec_results):
            assert abs(tr.score - vr.score) < 1e-5

    def test_search_by_vector_wrong_dimension_raises(self, tmp_store):
        """Passing a vector with wrong dimensions should raise ValueError."""
        bad_vector = np.random.rand(100).astype(np.float32)
        with pytest.raises(ValueError, match="query_vector must have shape"):
            tmp_store.search_by_vector(bad_vector)


# ===============================================================
# TESTS: search() with metadata filters
# ===============================================================

class TestSearchFilters:
    """Test that metadata filters narrow the search correctly."""

    def test_filter_returns_only_matching(self, tmp_store):
        """Filtering by category='tech' should only return tech records."""
        results = tmp_store.search(
            "learning algorithms",
            filters={"category": "tech"},
        )
        for r in results:
            assert r.record.metadata.get("category") == "tech", (
                f"Record {r.record.id} has category "
                f"'{r.record.metadata.get('category')}', expected 'tech'"
            )

    def test_filter_reduces_result_count(self, tmp_store):
        """Filtering should return fewer results than unfiltered search."""
        all_results = tmp_store.search("learning", top_k=10)
        filtered = tmp_store.search(
            "learning",
            top_k=10,
            filters={"category": "tech"},
        )
        assert len(filtered) < len(all_results)

    def test_filter_no_match_returns_empty(self, tmp_store):
        """Filtering with a non-existent value should return empty list."""
        results = tmp_store.search(
            "anything",
            filters={"category": "nonexistent_value"},
        )
        assert results == []

    def test_filter_multiple_criteria(self, tmp_store):
        """Multiple filters should AND together."""
        results = tmp_store.search(
            "learning",
            filters={"category": "tech", "topic": "ml"},
        )
        # Only the first record has both category=tech AND topic=ml
        assert len(results) == 1
        assert results[0].record.metadata["topic"] == "ml"

    def test_filter_with_animals_category(self, tmp_store):
        """Filtering by category='animals' should return cat/dog records."""
        results = tmp_store.search(
            "animals pets",
            filters={"category": "animals"},
        )
        assert len(results) == 2
        topics = {r.record.metadata["topic"] for r in results}
        assert topics == {"cats", "dogs"}


# ===============================================================
# TESTS: __len__
# ===============================================================

class TestLen:
    """Test the __len__ method."""

    def test_len_empty(self, empty_store):
        """len() of an empty store should be 0."""
        assert len(empty_store) == 0

    def test_len_matches_count(self, tmp_store):
        """len(store) should equal store.count()."""
        assert len(tmp_store) == tmp_store.count()
        assert len(tmp_store) == len(SCIENCE_TEXTS)


# ===============================================================
# TESTS: Invalid metric
# ===============================================================

class TestSearchInvalidMetric:
    """Test that unknown metrics are rejected."""

    def test_invalid_metric_raises_valueerror(self, tmp_store):
        """Using an unknown metric name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            tmp_store.search("test", metric="manhattan")

    def test_metric_name_is_normalized(self, tmp_store):
        """Metric names should be normalized to lowercase in SearchResult."""
        results = tmp_store.search("test", metric="COSINE")
        assert len(results) > 0
        assert all(r.metric == "cosine" for r in results)


class TestSearchTopKValidation:
    """Test that invalid top_k values are rejected clearly."""

    def test_search_invalid_top_k_raises_valueerror(self, tmp_store):
        """search() should reject non-positive top_k values."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            tmp_store.search("test", top_k=0)

    def test_search_by_vector_invalid_top_k_raises_valueerror(self, tmp_store):
        """search_by_vector() should reject non-positive top_k values."""
        query_vec = tmp_store.embedding_engine.encode("test")
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            tmp_store.search_by_vector(query_vec, top_k=-1)
