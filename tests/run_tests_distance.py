"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Test Runner for Distance Metrics                 ║
║  File: minivecdb/tests/run_tests_distance.py                  ║
║                                                               ║
║  This runs ALL tests without needing pytest installed.         ║
║  On your own machine, use: pytest test_distance_metrics.py -v ║
╚═══════════════════════════════════════════════════════════════╝
"""

import sys
import os
import numpy as np

# Add project root to path
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

passed = 0
failed = 0

def assert_approx(actual, expected, label, tol=1e-10):
    """Check if two values are approximately equal."""
    global passed, failed
    if abs(actual - expected) < tol:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label} — got {actual}, expected {expected}")
        failed += 1

def assert_true(condition, label):
    """Check if a condition is True."""
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label} — condition was False")
        failed += 1

def assert_raises(exception_type, func, label):
    """Check if a function raises the expected exception."""
    global passed, failed
    try:
        func()
        print(f"  ✗ {label} — no exception raised")
        failed += 1
    except exception_type:
        print(f"  ✓ {label}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {label} — wrong exception: {type(e).__name__}: {e}")
        failed += 1


# ── Test data ──
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])      # identical
c = np.array([3.0, 2.0, 1.0])      # different
z = np.array([0.0, 0.0, 0.0])      # zero
opp = np.array([-1.0, -2.0, -3.0]) # opposite
perp_a = np.array([1.0, 0.0])      # perpendicular pair
perp_b = np.array([0.0, 1.0])

database = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
])

print("=" * 60)
print("MiniVecDB — Distance Metrics Full Test Suite")
print("=" * 60)

# ═══════════════════════════════════════════
print("\n── COSINE SIMILARITY ──")
# ═══════════════════════════════════════════
assert_approx(cosine_similarity(a, b), 1.0,
    "Identical vectors → 1.0")

assert_approx(cosine_similarity(a, opp), -1.0,
    "Opposite vectors → -1.0")

assert_approx(cosine_similarity(perp_a, perp_b), 0.0,
    "Perpendicular vectors → 0.0")

assert_approx(cosine_similarity(a, z), 0.0,
    "Zero vector → 0.0 (no direction)")

result_ac = cosine_similarity(a, c)
assert_true(-1.0 <= result_ac <= 1.0,
    f"Result in [-1, 1] range: {result_ac:.4f}")

# Magnitude invariance
a2 = np.array([1.0, 2.0])
b2 = np.array([3.0, 4.0])
orig = cosine_similarity(a2, b2)
scaled = cosine_similarity(a2 * 100, b2 * 0.01)
assert_approx(orig, scaled,
    f"Magnitude invariance: cos(a,b)={orig:.6f} == cos(100a, 0.01b)={scaled:.6f}")

# Symmetry
assert_approx(cosine_similarity(a, c), cosine_similarity(c, a),
    "Symmetry: cos(a,c) == cos(c,a)")

# Dimension mismatch
assert_raises(ValueError, lambda: cosine_similarity(np.array([1,2]), np.array([1,2,3])),
    "Dimension mismatch raises ValueError")

# High-dimensional
np.random.seed(42)
hd_a = np.random.rand(384)
hd_b = np.random.rand(384)
hd_result = cosine_similarity(hd_a, hd_b)
assert_true(-1.0 <= hd_result <= 1.0,
    f"384-dim vectors work: score={hd_result:.4f}")

# ═══════════════════════════════════════════
print("\n── EUCLIDEAN DISTANCE ──")
# ═══════════════════════════════════════════
assert_approx(euclidean_distance(a, b), 0.0,
    "Identical vectors → distance 0.0")

assert_approx(euclidean_distance(perp_a, perp_b), np.sqrt(2),
    f"[1,0] to [0,1] → √2 = {np.sqrt(2):.4f}")

assert_true(euclidean_distance(a, c) >= 0.0,
    "Distance is non-negative")

assert_approx(euclidean_distance(a, c), euclidean_distance(c, a),
    "Symmetry: euc(a,c) == euc(c,a)")

# Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
p1 = np.array([0.0, 0.0])
p2 = np.array([1.0, 0.0])
p3 = np.array([1.0, 1.0])
assert_true(
    euclidean_distance(p1, p3) <= euclidean_distance(p1, p2) + euclidean_distance(p2, p3) + 1e-10,
    "Triangle inequality holds")

assert_raises(ValueError, lambda: euclidean_distance(np.array([1,2]), np.array([1,2,3])),
    "Dimension mismatch raises ValueError")

# ═══════════════════════════════════════════
print("\n── DOT PRODUCT SIMILARITY ──")
# ═══════════════════════════════════════════
assert_approx(dot_product_similarity(np.array([1.0,2.0,3.0]), np.array([4.0,5.0,6.0])), 32.0,
    "Known result: [1,2,3]·[4,5,6] = 32")

assert_approx(dot_product_similarity(a, z), 0.0,
    "Dot with zero vector → 0.0")

# Self-dot = squared norm
dot_self = dot_product_similarity(a, a)
norm_sq = np.linalg.norm(a) ** 2
assert_approx(dot_self, norm_sq,
    f"a·a = ‖a‖²: {dot_self:.4f} == {norm_sq:.4f}")

# Equals cosine when normalised
a_n = normalise_vector(a)
c_n = normalise_vector(c)
dot_norm = dot_product_similarity(a_n, c_n)
cos_orig = cosine_similarity(a, c)
assert_approx(dot_norm, cos_orig,
    f"Normalised dot = cosine: {dot_norm:.6f} == {cos_orig:.6f}")

assert_approx(dot_product_similarity(a, c), dot_product_similarity(c, a),
    "Symmetry: dot(a,c) == dot(c,a)")

# ═══════════════════════════════════════════
print("\n── BATCH OPERATIONS ──")
# ═══════════════════════════════════════════
query = np.array([1.0, 1.0, 0.0])

# Batch cosine vs single
batch_cos = batch_cosine_similarity(query, database)
for i, vec in enumerate(database):
    single = cosine_similarity(query, vec)
    assert_approx(batch_cos[i], single,
        f"Batch cosine[{i}] matches single: {batch_cos[i]:.4f}")

# Batch euclidean vs single
batch_euc = batch_euclidean_distance(query, database)
for i, vec in enumerate(database):
    single = euclidean_distance(query, vec)
    assert_approx(batch_euc[i], single,
        f"Batch euclidean[{i}] matches single: {batch_euc[i]:.4f}")

# Batch dot vs single
batch_dot = batch_dot_product(query, database)
for i, vec in enumerate(database):
    single = dot_product_similarity(query, vec)
    assert_approx(batch_dot[i], single,
        f"Batch dot[{i}] matches single: {batch_dot[i]:.4f}")

# Empty database
empty = np.array([]).reshape(0, 3)
assert_true(len(batch_cosine_similarity(query, empty)) == 0,
    "Empty database returns empty array (cosine)")
assert_true(len(batch_euclidean_distance(query, empty)) == 0,
    "Empty database returns empty array (euclidean)")
assert_true(len(batch_dot_product(query, empty)) == 0,
    "Empty database returns empty array (dot)")

# Output shape
assert_true(batch_cosine_similarity(query, database).shape == (4,),
    "Batch cosine output shape = (N,)")

# ═══════════════════════════════════════════
print("\n── NORMALISE VECTOR ──")
# ═══════════════════════════════════════════
normed = normalise_vector(a)
assert_approx(np.linalg.norm(normed), 1.0,
    "Normalised vector has magnitude 1.0")

assert_approx(cosine_similarity(a, normed), 1.0,
    "Normalised vector preserves direction")

zero_normed = normalise_vector(z)
assert_true(np.allclose(zero_normed, z),
    "Zero vector stays zero after normalisation")

# ═══════════════════════════════════════════
print("\n── METRIC REGISTRY ──")
# ═══════════════════════════════════════════
cos_metric = get_metric("cosine")
assert_true(cos_metric["higher_is_better"] is True,
    "Cosine: higher_is_better = True")
assert_true(callable(cos_metric["single"]),
    "Cosine: single function is callable")

euc_metric = get_metric("euclidean")
assert_true(euc_metric["higher_is_better"] is False,
    "Euclidean: higher_is_better = False")

dot_metric = get_metric("dot")
assert_true(dot_metric["higher_is_better"] is True,
    "Dot: higher_is_better = True")

assert_raises(ValueError, lambda: get_metric("manhattan"),
    "Unknown metric raises ValueError")

assert_true(get_metric("COSINE") == get_metric("cosine"),
    "Case-insensitive lookup works")

metrics = list_metrics()
assert_true(len(metrics) == 3,
    "list_metrics returns all 3 metrics")


# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)

if failed == 0:
    print("🎉 ALL TESTS PASSED! Day 2 module is rock solid.")
else:
    print(f"⚠️  {failed} test(s) failed. Review the output above.")
    sys.exit(1)
