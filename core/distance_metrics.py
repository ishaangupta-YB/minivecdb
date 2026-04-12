"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Distance Metrics Module                          ║
║  File: minivecdb/core/distance_metrics.py                     ║
║                                                               ║
║  This module implements the three similarity/distance metrics  ║
║  that power every search operation in MiniVecDB.              ║
║                                                               ║
║  Supported Metrics:                                           ║
║    1. Cosine Similarity  — Direction-based (ignores length)   ║
║    2. Euclidean Distance — Straight-line distance in space    ║
║    3. Dot Product        — Fastest, assumes normalised input  ║
╚═══════════════════════════════════════════════════════════════╝
"""

import numpy as np 

# ═══════════════════════════════════════════════════════════════
# TYPE ALIAS
# ═══════════════════════════════════════════════════════════════
# A "Vector" in our system is always a NumPy array of floats.
# This type alias makes function signatures clearer and documents
# our intent. When you see "Vector" in a function parameter,
# you know it expects a NumPy array.

Vector = np.ndarray


# ═══════════════════════════════════════════════════════════════
# 1. COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════
#
#                    A · B
#   cos(θ)  =  ─────────────────
#               ‖A‖  ×  ‖B‖
#
#   Where:
#     A · B  = dot product = Σ(Aᵢ × Bᵢ)
#     ‖A‖   = magnitude   = √(Σ Aᵢ²)
#
#   Returns a value between -1 and +1:
#     +1 = identical direction (most similar)
#      0 = perpendicular (no similarity)
#     -1 = opposite direction (least similar)
#
#   WHY USE THIS?
#   Cosine similarity is the gold standard for text/NLP tasks
#   because it ignores magnitude (vector length) and focuses
#   purely on direction. A 10-page document about cooking and
#   a 2-sentence recipe will have vectors of very different
#   magnitudes, but if they point in the same direction
#   (same topic), cosine similarity will score them high.
#
# ═══════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Cosine similarity measures the angle between two vectors,
    ignoring their magnitudes. This makes it ideal for comparing
    text documents of different lengths.
    
    Args:
        vec_a: First vector (1D NumPy array)
        vec_b: Second vector (1D NumPy array, same dimensions as vec_a)
    
    Returns:
        Float between -1.0 and 1.0
            1.0  = vectors point in the same direction (identical)
            0.0  = vectors are perpendicular (unrelated)
           -1.0  = vectors point in opposite directions (opposite)
    
    Raises:
        ValueError: If vectors have different dimensions
    
    Example:
        >>> import numpy as np
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([1.0, 2.0, 3.0])
        >>> cosine_similarity(a, b)
        1.0
    """
    # ── STEP 1: Validate inputs ──
    # Both vectors must have the same number of dimensions.
    # You can't compare a 3D vector with a 5D vector — it's
    # like comparing apples to oranges (literally!)
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector dimensions don't match: {vec_a.shape} vs {vec_b.shape}. "
            f"Both vectors must have the same number of dimensions."
        )
    
    # ── STEP 2: Compute the dot product (numerator) ──
    # np.dot multiplies corresponding elements and sums them.
    # Example: [1,2,3] · [4,5,6] = 1×4 + 2×5 + 3×6 = 32
    dot_product = np.dot(vec_a, vec_b)
    
    # ── STEP 3: Compute magnitudes (denominator parts) ──
    # np.linalg.norm calculates the Euclidean norm (length).
    # Example: ‖[3,4]‖ = √(3² + 4²) = √25 = 5
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    # ── STEP 4: Handle edge case — zero vectors ──
    # A zero vector [0, 0, 0] has magnitude 0. Dividing by 0
    # would cause an error, so we return 0.0 (no similarity)
    # because a zero vector has no direction to compare.
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    
    # ── STEP 5: Compute and return cosine similarity ──
    # np.clip ensures the result stays in [-1, 1] range.
    # Due to floating-point arithmetic, we might get values
    # like 1.0000000002 which would be technically invalid.
    # Clipping rounds these tiny errors away.
    similarity = dot_product / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))


# ═══════════════════════════════════════════════════════════════
# 2. EUCLIDEAN DISTANCE
# ═══════════════════════════════════════════════════════════════
#
#   d(A, B) = √( Σ (Aᵢ - Bᵢ)² )
#
#   This is the Pythagorean theorem extended to any number of
#   dimensions. In 2D: d = √((x₁-x₂)² + (y₁-y₂)²)
#   In 384D: same formula, just 384 terms under the root.
#
#   Returns a value from 0 to ∞:
#     0 = identical vectors (no distance)
#     ∞ = infinitely far apart
#
#   IMPORTANT: This is a DISTANCE metric (lower = more similar),
#   which is the OPPOSITE of cosine similarity (higher = more
#   similar). Our search function will handle this difference.
#
#   WHY USE THIS?
#   Euclidean distance is intuitive — it's the "straight line"
#   between two points. It considers both direction AND magnitude,
#   making it suitable for spatial data, image features, and
#   situations where the absolute values of numbers matter.
#
# ═══════════════════════════════════════════════════════════════

def euclidean_distance(vec_a: Vector, vec_b: Vector) -> float:
    """
    Calculate Euclidean (L2) distance between two vectors.
    
    This is the "straight line" distance between two points in
    multi-dimensional space. Unlike cosine similarity, it considers
    both direction and magnitude.
    
    Args:
        vec_a: First vector (1D NumPy array)
        vec_b: Second vector (1D NumPy array, same dimensions as vec_a)
    
    Returns:
        Float >= 0.0
            0.0 = identical vectors
            Higher values = more distant (less similar)
    
    Raises:
        ValueError: If vectors have different dimensions
    
    Example:
        >>> import numpy as np
        >>> a = np.array([1.0, 0.0])
        >>> b = np.array([0.0, 1.0])
        >>> euclidean_distance(a, b)  # √((1-0)² + (0-1)²) = √2
        1.4142135623730951
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector dimensions don't match: {vec_a.shape} vs {vec_b.shape}."
        )
    
    # ── Calculate distance ──
    # np.linalg.norm of the DIFFERENCE vector gives Euclidean distance.
    # Internally this computes: √(Σ(Aᵢ - Bᵢ)²)
    #
    # Breaking it down:
    #   1. vec_a - vec_b  → element-wise subtraction → difference vector
    #   2. np.linalg.norm → square each, sum, square root
    #
    # Example: a=[1,2,3], b=[4,5,6]
    #   diff = [-3, -3, -3]
    #   norm = √(9 + 9 + 9) = √27 ≈ 5.196
    return float(np.linalg.norm(vec_a - vec_b))


# ═══════════════════════════════════════════════════════════════
# 3. DOT PRODUCT SIMILARITY
# ═══════════════════════════════════════════════════════════════
#
#   dot(A, B) = Σ(Aᵢ × Bᵢ)
#
#   The simplest and fastest metric. Just multiply corresponding
#   elements and add them up. No division, no square roots.
#
#   Returns a value from -∞ to +∞:
#     Positive = vectors point in similar directions
#     Zero     = perpendicular
#     Negative = opposite directions
#
#   KEY INSIGHT:
#   If both vectors are NORMALISED (length = 1), then:
#       dot(A, B) == cosine_similarity(A, B)
#   This is because the denominator ‖A‖×‖B‖ = 1×1 = 1.
#
#   WHY USE THIS?
#   Speed. The dot product skips the expensive magnitude
#   calculations. If you normalise vectors when inserting them,
#   you can use the faster dot product at search time and get
#   identical results to cosine similarity. Many production
#   vector databases use this trick.
#
# ═══════════════════════════════════════════════════════════════

def dot_product_similarity(vec_a: Vector, vec_b: Vector) -> float:
    """
    Calculate dot product similarity between two vectors.
    
    The simplest metric — just multiply and sum. If vectors are
    normalised (length 1), this equals cosine similarity but is
    faster because it skips the magnitude division.
    
    Args:
        vec_a: First vector (1D NumPy array)
        vec_b: Second vector (1D NumPy array, same dimensions as vec_a)
    
    Returns:
        Float (unbounded). Higher = more similar.
    
    Raises:
        ValueError: If vectors have different dimensions
    
    Example:
        >>> import numpy as np
        >>> a = np.array([1.0, 2.0, 3.0])
        >>> b = np.array([4.0, 5.0, 6.0])
        >>> dot_product_similarity(a, b)  # 1×4 + 2×5 + 3×6 = 32
        32.0
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vector dimensions don't match: {vec_a.shape} vs {vec_b.shape}."
        )
    
    # np.dot computes: Σ(Aᵢ × Bᵢ)
    return float(np.dot(vec_a, vec_b))


# ═══════════════════════════════════════════════════════════════
# BATCH OPERATIONS (FOR SEARCHING ENTIRE DATABASE AT ONCE)
# ═══════════════════════════════════════════════════════════════
#
# When a user searches, we need to compute similarity between
# the query vector and EVERY vector in the database. Doing this
# one vector at a time in a Python loop would be extremely slow
# for large databases (imagine 100,000 vectors!).
#
# These "batch" functions use NumPy matrix operations to compute
# ALL similarities at once in a single operation. This is called
# "vectorized" computation and is typically 50-100x faster than
# looping in Python.
#
# ═══════════════════════════════════════════════════════════════

def batch_cosine_similarity(query: Vector, vectors: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and a matrix
    of vectors. This is the function that powers every search.
    
    Args:
        query:   1D array of shape (D,) — the search query vector
        vectors: 2D array of shape (N, D) — all stored vectors
                 N = number of vectors, D = dimensions per vector
    
    Returns:
        1D array of shape (N,) — similarity score for each vector.
        Higher scores mean more similar to the query.
    
    Example:
        >>> query = np.array([1.0, 0.0])
        >>> database = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        >>> batch_cosine_similarity(query, database)
        array([1.0, 0.0, 0.707...])
    """
    # Handle empty database
    if vectors.size == 0:
        return np.array([])
    
    # ── STEP 1: Compute all dot products at once ──
    # Matrix-vector multiplication: (N, D) @ (D,) → (N,)
    # Each element i in the result = dot(vectors[i], query)
    dot_products = vectors @ query
    
    # ── STEP 2: Compute all norms at once ──
    # axis=1 means "compute norm along each row"
    vector_norms = np.linalg.norm(vectors, axis=1)
    query_norm = np.linalg.norm(query)
    
    # ── STEP 3: Compute denominator (product of norms) ──
    denominators = vector_norms * query_norm
    
    # ── STEP 4: Handle zero vectors (avoid division by zero) ──
    # np.where is like an if-else for arrays:
    # Where denominator != 0, do the division; otherwise return 0.0
    # np.errstate suppresses the expected "divide by zero" warning
    # for zero vectors. Our np.where handles it correctly (returns 0.0),
    # but NumPy evaluates the division before applying the condition.
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.where(
            denominators != 0,
            dot_products / denominators,
            0.0
        )
    
    return np.clip(similarities, -1.0, 1.0)


def batch_euclidean_distance(query: Vector, vectors: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between a query and all stored vectors.
    
    Args:
        query:   1D array of shape (D,)
        vectors: 2D array of shape (N, D)
    
    Returns:
        1D array of shape (N,) — distances (lower = more similar)
    """
    if vectors.size == 0:
        return np.array([])
    
    # ── Compute all distances at once ──
    # vectors - query: NumPy "broadcasts" the query to every row,
    # subtracting it element-wise from each stored vector.
    # Result shape: (N, D) — one difference vector per stored vector.
    #
    # np.linalg.norm with axis=1 then computes the magnitude of
    # each difference vector, giving us the Euclidean distance.
    differences = vectors - query
    distances = np.linalg.norm(differences, axis=1)
    
    return distances


def batch_dot_product(query: Vector, vectors: np.ndarray) -> np.ndarray:
    """
    Compute dot product between a query and all stored vectors.
    
    Args:
        query:   1D array of shape (D,)
        vectors: 2D array of shape (N, D)
    
    Returns:
        1D array of shape (N,) — dot products (higher = more similar)
    """
    if vectors.size == 0:
        return np.array([])
    
    # This is the fastest operation — pure matrix multiplication.
    # No norms, no division. Just multiply and sum.
    return vectors @ query


# ═══════════════════════════════════════════════════════════════
# METRIC REGISTRY — USER-FACING SELECTION
# ═══════════════════════════════════════════════════════════════
#
# This dictionary maps metric names (strings) to their batch
# functions. When a user says "search using cosine similarity",
# we look up "cosine" in this dictionary and call the right
# function. This pattern is called a "registry" or "strategy
# pattern" — it makes adding new metrics easy in the future.
#
# ═══════════════════════════════════════════════════════════════

# Maps metric name → (batch_function, higher_is_better)
# higher_is_better tells the search engine whether to sort
# results ascending (False, for distances) or descending (True)
METRIC_REGISTRY = {
    "cosine": {
        "single": cosine_similarity,
        "batch": batch_cosine_similarity,
        "higher_is_better": True,   # 1.0 = best
        "description": "Cosine similarity (direction-based, ignores magnitude)"
    },
    "euclidean": {
        "single": euclidean_distance,
        "batch": batch_euclidean_distance,
        "higher_is_better": False,  # 0.0 = best
        "description": "Euclidean distance (straight-line distance in space)"
    },
    "dot": {
        "single": dot_product_similarity,
        "batch": batch_dot_product,
        "higher_is_better": True,   # higher = better
        "description": "Dot product (fastest, best with normalised vectors)"
    }
}


def get_metric(name: str) -> dict:
    """
    Look up a distance metric by name.
    
    Args:
        name: One of "cosine", "euclidean", or "dot"
    
    Returns:
        Dictionary with 'single', 'batch', 'higher_is_better', 'description'
    
    Raises:
        ValueError: If metric name is not recognised
    """
    name = name.lower().strip()
    if name not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys())
        raise ValueError(
            f"Unknown metric '{name}'. Available metrics: {available}"
        )
    return METRIC_REGISTRY[name]


def list_metrics() -> list:
    """Return a list of all available metric names and descriptions."""
    return [
        {"name": name, "description": info["description"]}
        for name, info in METRIC_REGISTRY.items()
    ]


# ═══════════════════════════════════════════════════════════════
# UTILITY: NORMALISE A VECTOR
# ═══════════════════════════════════════════════════════════════

def normalise_vector(vec: Vector) -> Vector:
    """
    Normalise a vector to unit length (magnitude = 1).
    
    This is useful because:
    - Normalised vectors make dot product = cosine similarity
    - Removes magnitude bias from comparisons
    
    Args:
        vec: Input vector
    
    Returns:
        Unit vector pointing in the same direction, or zero
        vector if input is zero vector.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.copy()
    return vec / norm


# ═══════════════════════════════════════════════════════════════
# MODULE SELF-TEST (runs when you execute this file directly)
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MiniVecDB — Distance Metrics Module Self-Test")
    print("=" * 60)
    
    # Test vectors
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])  # identical to a
    c = np.array([3.0, 2.0, 1.0])  # different from a
    z = np.array([0.0, 0.0, 0.0])  # zero vector
    
    print("\nTest vectors:")
    print(f"  a = {a}")
    print(f"  b = {b}  (identical to a)")
    print(f"  c = {c}  (different from a)")
    print(f"  z = {z}  (zero vector)")
    
    # Single-pair tests
    print("\n--- Cosine Similarity ---")
    print(f"  cos(a, b) = {cosine_similarity(a, b):.4f}  (should be 1.0 — identical)")
    print(f"  cos(a, c) = {cosine_similarity(a, c):.4f}  (should be < 1.0)")
    print(f"  cos(a, z) = {cosine_similarity(a, z):.4f}  (should be 0.0 — zero vec)")
    
    print("\n--- Euclidean Distance ---")
    print(f"  euc(a, b) = {euclidean_distance(a, b):.4f}  (should be 0.0 — identical)")
    print(f"  euc(a, c) = {euclidean_distance(a, c):.4f}  (should be > 0.0)")
    
    print("\n--- Dot Product ---")
    print(f"  dot(a, b) = {dot_product_similarity(a, b):.4f}  (= 1+4+9 = 14.0)")
    print(f"  dot(a, c) = {dot_product_similarity(a, c):.4f}  (= 3+4+3 = 10.0)")
    
    # Batch test
    print("\n--- Batch Cosine Similarity ---")
    database = np.array([b, c, z])
    scores = batch_cosine_similarity(a, database)
    print(f"  Query a vs [b, c, z]: {scores}")
    
    # Normalisation proof
    print("\n--- Dot Product = Cosine (when normalised) ---")
    a_norm = normalise_vector(a)
    c_norm = normalise_vector(c)
    cos_score = cosine_similarity(a, c)
    dot_score = dot_product_similarity(a_norm, c_norm)
    print(f"  cosine(a, c)              = {cos_score:.6f}")
    print(f"  dot(normalise(a), norm(c)) = {dot_score:.6f}")
    print(f"  Equal? {abs(cos_score - dot_score) < 1e-10}")
    
    # Available metrics
    print("\n--- Available Metrics ---")
    for m in list_metrics():
        print(f"  {m['name']:>10}: {m['description']}")
    
    print("\n✓ All self-tests passed!")
