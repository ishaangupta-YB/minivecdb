"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Unit Tests for Distance Metrics                  ║
║  File: minivecdb/tests/test_distance_metrics.py               ║
║                                                               ║
║  Run with: pytest minivecdb/tests/test_distance_metrics.py -v ║
║     or:    python -m pytest tests/test_distance_metrics.py -v ║
╚═══════════════════════════════════════════════════════════════╝

WHAT IS PYTEST?
    pytest is a testing framework for Python. It lets you write
    small functions that check if your code works correctly.
    Each test function starts with "test_" and uses "assert"
    statements to verify expected results.

    If all asserts pass → test passes (green ✓)
    If any assert fails → test fails (red ✗) with a clear message

WHY WRITE TESTS?
    1. They PROVE your code works (not just "it looks right")
    2. They catch bugs early before they cause bigger problems
    3. They let you refactor (rewrite) code confidently
    4. They serve as living documentation of how code behaves
"""

import numpy as np
import pytest

# ── Import the module we're testing ──
# We import from our project package. The "sys.path" line ensures
# Python can find our package from the test directory.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.distance_metrics import (
    cosine_similarity,
    euclidean_distance,
    dot_product_similarity,
    batch_cosine_similarity,
    batch_euclidean_distance,
    batch_dot_product,
    normalise_vector,
    get_metric,
    list_metrics,
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES — Reusable test data
# ═══════════════════════════════════════════════════════════════
# A "fixture" in pytest is a function decorated with @pytest.fixture
# that provides test data. Tests can request fixtures by name in
# their parameter list, and pytest automatically calls the fixture
# and passes the result. This avoids repeating setup code.

@pytest.fixture
def vec_a():
    """A simple 3D vector."""
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def vec_b_identical():
    """Identical to vec_a — tests for perfect similarity."""
    return np.array([1.0, 2.0, 3.0])

@pytest.fixture
def vec_c_different():
    """Different from vec_a — tests for partial similarity."""
    return np.array([3.0, 2.0, 1.0])

@pytest.fixture
def vec_zero():
    """Zero vector — edge case for division by zero."""
    return np.array([0.0, 0.0, 0.0])

@pytest.fixture
def vec_opposite():
    """Opposite direction of vec_a — tests for -1 similarity."""
    return np.array([-1.0, -2.0, -3.0])

@pytest.fixture
def vec_perpendicular():
    """Perpendicular to [1, 0] in 2D — tests for 0 similarity."""
    return np.array([0.0, 1.0])

@pytest.fixture
def sample_database():
    """A small database matrix for batch operation tests."""
    return np.array([
        [1.0, 0.0, 0.0],  # points along x
        [0.0, 1.0, 0.0],  # points along y
        [0.0, 0.0, 1.0],  # points along z
        [1.0, 1.0, 1.0],  # diagonal
    ])


# ═══════════════════════════════════════════════════════════════
# TESTS: COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    """Group all cosine similarity tests together."""
    
    def test_identical_vectors_return_one(self, vec_a, vec_b_identical):
        """Two identical vectors should have cosine similarity of exactly 1.0."""
        result = cosine_similarity(vec_a, vec_b_identical)
        assert result == pytest.approx(1.0, abs=1e-10)
    
    def test_opposite_vectors_return_negative_one(self, vec_a, vec_opposite):
        """Opposite vectors should have cosine similarity of -1.0."""
        result = cosine_similarity(vec_a, vec_opposite)
        assert result == pytest.approx(-1.0, abs=1e-10)
    
    def test_perpendicular_vectors_return_zero(self):
        """Perpendicular vectors should have cosine similarity of 0.0."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_zero_vector_returns_zero(self, vec_a, vec_zero):
        """A zero vector has no direction, so similarity should be 0.0."""
        assert cosine_similarity(vec_a, vec_zero) == 0.0
        assert cosine_similarity(vec_zero, vec_a) == 0.0
        assert cosine_similarity(vec_zero, vec_zero) == 0.0
    
    def test_result_between_minus_one_and_one(self, vec_a, vec_c_different):
        """Result should always be in [-1, 1] range."""
        result = cosine_similarity(vec_a, vec_c_different)
        assert -1.0 <= result <= 1.0
    
    def test_magnitude_invariance(self):
        """Cosine similarity should NOT change when vectors are scaled."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        a_scaled = a * 100   # 100x longer but same direction
        b_scaled = b * 0.01  # 100x shorter but same direction
        
        original = cosine_similarity(a, b)
        scaled = cosine_similarity(a_scaled, b_scaled)
        assert original == pytest.approx(scaled, abs=1e-10)
    
    def test_symmetry(self, vec_a, vec_c_different):
        """cos(a, b) should equal cos(b, a) — order shouldn't matter."""
        assert cosine_similarity(vec_a, vec_c_different) == pytest.approx(
            cosine_similarity(vec_c_different, vec_a), abs=1e-10
        )
    
    def test_dimension_mismatch_raises_error(self):
        """Different-dimensional vectors should raise ValueError."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="dimensions don't match"):
            cosine_similarity(a, b)
    
    def test_high_dimensional_vectors(self):
        """Should work correctly with 384-dimensional vectors (real use case)."""
        np.random.seed(42)
        a = np.random.rand(384)
        b = np.random.rand(384)
        result = cosine_similarity(a, b)
        assert -1.0 <= result <= 1.0
        assert isinstance(result, float)


# ═══════════════════════════════════════════════════════════════
# TESTS: EUCLIDEAN DISTANCE
# ═══════════════════════════════════════════════════════════════

class TestEuclideanDistance:
    """Group all Euclidean distance tests together."""
    
    def test_identical_vectors_return_zero(self, vec_a, vec_b_identical):
        """Distance between identical vectors should be 0.0."""
        result = euclidean_distance(vec_a, vec_b_identical)
        assert result == pytest.approx(0.0, abs=1e-10)
    
    def test_known_distance(self):
        """Test against a hand-calculated result."""
        # Distance from [1,0] to [0,1] = √((1-0)² + (0-1)²) = √2
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        result = euclidean_distance(a, b)
        assert result == pytest.approx(np.sqrt(2), abs=1e-10)
    
    def test_always_non_negative(self, vec_a, vec_c_different):
        """Distance should never be negative."""
        result = euclidean_distance(vec_a, vec_c_different)
        assert result >= 0.0
    
    def test_symmetry(self, vec_a, vec_c_different):
        """euc(a, b) should equal euc(b, a)."""
        assert euclidean_distance(vec_a, vec_c_different) == pytest.approx(
            euclidean_distance(vec_c_different, vec_a), abs=1e-10
        )
    
    def test_triangle_inequality(self):
        """d(a,c) <= d(a,b) + d(b,c) — fundamental property of distances."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([1.0, 1.0])
        
        d_ac = euclidean_distance(a, c)
        d_ab = euclidean_distance(a, b)
        d_bc = euclidean_distance(b, c)
        
        assert d_ac <= d_ab + d_bc + 1e-10  # small epsilon for float errors
    
    def test_dimension_mismatch_raises_error(self):
        """Different-dimensional vectors should raise ValueError."""
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="dimensions don't match"):
            euclidean_distance(a, b)


# ═══════════════════════════════════════════════════════════════
# TESTS: DOT PRODUCT SIMILARITY
# ═══════════════════════════════════════════════════════════════

class TestDotProductSimilarity:
    """Group all dot product tests together."""
    
    def test_known_result(self):
        """Test against hand-calculated dot product."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        # 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
        assert dot_product_similarity(a, b) == pytest.approx(32.0)
    
    def test_zero_vector(self, vec_a, vec_zero):
        """Dot product with zero vector should be 0.0."""
        assert dot_product_similarity(vec_a, vec_zero) == 0.0
    
    def test_self_dot_equals_squared_norm(self, vec_a):
        """a · a should equal ‖a‖². This is a mathematical identity."""
        dot = dot_product_similarity(vec_a, vec_a)
        norm_squared = np.linalg.norm(vec_a) ** 2
        assert dot == pytest.approx(norm_squared, abs=1e-10)
    
    def test_equals_cosine_when_normalised(self, vec_a, vec_c_different):
        """When both vectors are normalised, dot product = cosine similarity."""
        a_norm = normalise_vector(vec_a)
        c_norm = normalise_vector(vec_c_different)
        
        dot_result = dot_product_similarity(a_norm, c_norm)
        cos_result = cosine_similarity(vec_a, vec_c_different)
        
        assert dot_result == pytest.approx(cos_result, abs=1e-10)
    
    def test_symmetry(self, vec_a, vec_c_different):
        """dot(a, b) should equal dot(b, a)."""
        assert dot_product_similarity(vec_a, vec_c_different) == pytest.approx(
            dot_product_similarity(vec_c_different, vec_a), abs=1e-10
        )


# ═══════════════════════════════════════════════════════════════
# TESTS: BATCH OPERATIONS
# ═══════════════════════════════════════════════════════════════

class TestBatchOperations:
    """Test that batch operations match single-pair operations."""
    
    def test_batch_cosine_matches_single(self, sample_database):
        """Batch cosine results should match individual calculations."""
        query = np.array([1.0, 1.0, 0.0])
        batch_results = batch_cosine_similarity(query, sample_database)
        
        for i, vec in enumerate(sample_database):
            single_result = cosine_similarity(query, vec)
            assert batch_results[i] == pytest.approx(single_result, abs=1e-10), \
                f"Mismatch at index {i}: batch={batch_results[i]}, single={single_result}"
    
    def test_batch_euclidean_matches_single(self, sample_database):
        """Batch euclidean results should match individual calculations."""
        query = np.array([1.0, 1.0, 0.0])
        batch_results = batch_euclidean_distance(query, sample_database)
        
        for i, vec in enumerate(sample_database):
            single_result = euclidean_distance(query, vec)
            assert batch_results[i] == pytest.approx(single_result, abs=1e-10)
    
    def test_batch_dot_matches_single(self, sample_database):
        """Batch dot product results should match individual calculations."""
        query = np.array([1.0, 1.0, 0.0])
        batch_results = batch_dot_product(query, sample_database)
        
        for i, vec in enumerate(sample_database):
            single_result = dot_product_similarity(query, vec)
            assert batch_results[i] == pytest.approx(single_result, abs=1e-10)
    
    def test_batch_with_empty_database(self):
        """Batch operations should handle empty databases gracefully."""
        query = np.array([1.0, 2.0, 3.0])
        empty = np.array([]).reshape(0, 3)
        
        assert len(batch_cosine_similarity(query, empty)) == 0
        assert len(batch_euclidean_distance(query, empty)) == 0
        assert len(batch_dot_product(query, empty)) == 0
    
    def test_batch_output_shape(self, sample_database):
        """Batch operations should return arrays of length N."""
        query = np.array([1.0, 1.0, 0.0])
        n = len(sample_database)
        
        assert batch_cosine_similarity(query, sample_database).shape == (n,)
        assert batch_euclidean_distance(query, sample_database).shape == (n,)
        assert batch_dot_product(query, sample_database).shape == (n,)


# ═══════════════════════════════════════════════════════════════
# TESTS: NORMALISE VECTOR
# ═══════════════════════════════════════════════════════════════

class TestNormaliseVector:
    """Test the vector normalisation utility."""
    
    def test_normalised_has_unit_length(self, vec_a):
        """Normalised vector should have magnitude 1.0."""
        result = normalise_vector(vec_a)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-10)
    
    def test_normalised_preserves_direction(self, vec_a):
        """Normalised vector should point in the same direction."""
        result = normalise_vector(vec_a)
        cos_sim = cosine_similarity(vec_a, result)
        assert cos_sim == pytest.approx(1.0, abs=1e-10)
    
    def test_zero_vector_stays_zero(self, vec_zero):
        """Normalising a zero vector should return a zero vector."""
        result = normalise_vector(vec_zero)
        assert np.allclose(result, vec_zero)


# ═══════════════════════════════════════════════════════════════
# TESTS: METRIC REGISTRY
# ═══════════════════════════════════════════════════════════════

class TestMetricRegistry:
    """Test the metric lookup and listing system."""
    
    def test_get_cosine_metric(self):
        """Should return the cosine metric configuration."""
        metric = get_metric("cosine")
        assert metric["higher_is_better"] is True
        assert callable(metric["single"])
        assert callable(metric["batch"])
    
    def test_get_euclidean_metric(self):
        """Should return euclidean as lower-is-better."""
        metric = get_metric("euclidean")
        assert metric["higher_is_better"] is False
    
    def test_get_dot_metric(self):
        """Should return the dot product metric."""
        metric = get_metric("dot")
        assert metric["higher_is_better"] is True
    
    def test_unknown_metric_raises_error(self):
        """Requesting an unknown metric should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("manhattan")
    
    def test_case_insensitive_lookup(self):
        """Metric lookup should be case-insensitive."""
        assert get_metric("COSINE") == get_metric("cosine")
        assert get_metric("Euclidean") == get_metric("euclidean")
    
    def test_list_metrics_returns_all_three(self):
        """list_metrics should return all three available metrics."""
        metrics = list_metrics()
        names = [m["name"] for m in metrics]
        assert "cosine" in names
        assert "euclidean" in names
        assert "dot" in names
        assert len(metrics) == 3
