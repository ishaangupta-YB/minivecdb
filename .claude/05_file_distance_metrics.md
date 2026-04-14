# MiniVecDB — File: core/distance_metrics.py

> **Location**: `minivecdb/core/distance_metrics.py`
> **Lines**: 519 | **Size**: 21.8 KB
> **Purpose**: Implements the three similarity/distance metrics that power vector search — **built from scratch**

---

## Why This File Exists

This is one of the files that makes MiniVecDB a "vector database built from scratch." Instead of using a library like FAISS for similarity computation, every metric is implemented directly using NumPy operations. This file is essential for the DBMS course because it demonstrates the *mathematical foundation* of vector search.

---

## Concepts

### What Is a Similarity Metric?
A similarity metric is a function that takes two vectors and returns a number indicating how "similar" they are. Different metrics capture different notions of similarity:

| Metric | Measures | Higher = ? | Best For |
|--------|----------|-----------|----------|
| Cosine | Direction angle | More similar | Text (ignores document length) |
| Euclidean | Straight-line distance | Less similar | Spatial data, images |
| Dot Product | Weighted alignment | More similar | Speed (pre-normalized vectors) |

---

## Functions — Detailed Breakdown

### Single-Pair Functions

#### `cosine_similarity(vec_a, vec_b) → float`
**Lines 56–118** | The gold standard for text similarity.

**Formula**: `cos(θ) = A·B / (‖A‖ × ‖B‖)`

**Algorithm step-by-step**:
1. **Validate**: Check `vec_a.shape == vec_b.shape` (can't compare 3D with 5D)
2. **Dot product**: `np.dot(vec_a, vec_b)` — multiply element-wise, sum results
3. **Magnitudes**: `np.linalg.norm(vec_a)` and `np.linalg.norm(vec_b)` — Euclidean lengths
4. **Zero check**: If either magnitude is 0 (zero vector), return 0.0 (no direction to compare)
5. **Divide and clip**: `dot_product / (norm_a * norm_b)`, then `np.clip(result, -1, 1)` to fix floating-point rounding errors

**Why cosine for text?** A 10-page document about cooking and a 2-sentence recipe have very different vector magnitudes, but if they point in the same direction (same topic), cosine will score them high. It ignores "length" and focuses on "direction."

**Returns**: Float in `[-1, 1]` where `1.0` = identical direction, `0.0` = perpendicular, `-1.0` = opposite.

---

#### `euclidean_distance(vec_a, vec_b) → float`
**Lines 147–190** | Straight-line distance in multi-dimensional space.

**Formula**: `d(A,B) = √(Σ(Aᵢ - Bᵢ)²)` — the Pythagorean theorem extended to 384 dimensions.

**Algorithm**:
1. **Validate**: Shape check
2. **Compute**: `np.linalg.norm(vec_a - vec_b)` — internally does `√(Σ(differences²))`

**Key difference from cosine**: This is a *distance* (lower = more similar), not a *similarity* (higher = more similar). The search engine handles this difference when sorting results.

**Returns**: Float in `[0, ∞)` where `0.0` = identical vectors.

---

#### `dot_product_similarity(vec_a, vec_b) → float`
**Lines 221–252** | The simplest and fastest metric.

**Formula**: `dot(A,B) = Σ(Aᵢ × Bᵢ)` — just multiply and sum.

**Algorithm**:
1. **Validate**: Shape check
2. **Compute**: `np.dot(vec_a, vec_b)` — no division, no square roots

**Key insight**: If both vectors are normalised (length = 1), then `dot(A,B) == cosine(A,B)`. This is because the denominator `‖A‖×‖B‖ = 1×1 = 1`. Many production databases pre-normalize vectors at insert time and use dot product at search time for speed.

**Returns**: Float in `(-∞, ∞)` where higher = more similar.

---

### Batch Functions (Powers Actual Search)

These are the functions that actually get called during search. They compute similarity between one query vector and ALL stored vectors at once using NumPy matrix operations. This is **50–100x faster** than a Python loop.

#### `batch_cosine_similarity(query, vectors) → np.ndarray`
**Lines 271–321**

**Input**: `query` shape `(384,)`, `vectors` shape `(N, 384)`
**Output**: `(N,)` — one score per stored vector

**Algorithm**:
1. `dot_products = vectors @ query` — Matrix-vector multiplication, result shape `(N,)`. Each element = dot product of one stored vector with the query.
2. `vector_norms = np.linalg.norm(vectors, axis=1)` — Norm of each row, shape `(N,)`
3. `query_norm = np.linalg.norm(query)` — Scalar
4. `denominators = vector_norms * query_norm` — Shape `(N,)`
5. `similarities = np.where(denominators != 0, dot_products / denominators, 0.0)` — Avoid division by zero
6. `np.clip(similarities, -1, 1)` — Fix floating-point errors

**Why this is fast**: `vectors @ query` is a single CPU/BLAS operation that processes all N vectors simultaneously. A Python loop would call `np.dot()` N times with Python overhead between calls.

#### `batch_euclidean_distance(query, vectors) → np.ndarray`
**Lines 324–348**

```python
differences = vectors - query       # Broadcasting: (N,384) - (384,) → (N,384)
distances = np.linalg.norm(differences, axis=1)  # (N,)
```

NumPy "broadcasts" the query to every row automatically.

#### `batch_dot_product(query, vectors) → np.ndarray`
**Lines 351–367**

```python
return vectors @ query  # (N,384) @ (384,) → (N,)
```

The fastest — pure matrix multiplication with no extra work.

---

### Metric Registry (Strategy Pattern)
**Lines 385–434**

```python
METRIC_REGISTRY = {
    "cosine":    {"single": cosine_similarity,    "batch": batch_cosine_similarity,    "higher_is_better": True},
    "euclidean": {"single": euclidean_distance,   "batch": batch_euclidean_distance,   "higher_is_better": False},
    "dot":       {"single": dot_product_similarity,"batch": batch_dot_product,          "higher_is_better": True},
}
```

**What it is**: A dictionary mapping metric names to their functions and sorting direction.

**Why?** This is the **Strategy Pattern** — when a user says "search using cosine", the code does:
```python
metric_info = get_metric("cosine")
scores = metric_info["batch"](query, vectors)
if metric_info["higher_is_better"]:
    sorted_idx = np.argsort(scores)[::-1]  # descending
else:
    sorted_idx = np.argsort(scores)         # ascending
```

Adding a new metric in the future just means adding one entry to this dict.

#### `get_metric(name) → dict`
Looks up a metric by name, raises `ValueError` if unknown.

#### `list_metrics() → list`
Returns all available metrics with descriptions (used by CLI `--help`).

---

### Utility: `normalise_vector(vec) → Vector`
**Lines 441–459**

Divides a vector by its magnitude to make it unit length (magnitude = 1). Used when you want dot product to equal cosine similarity.

```python
norm = np.linalg.norm(vec)
if norm == 0: return vec.copy()  # Zero vector stays zero
return vec / norm
```

---

## Type Alias

```python
Vector = np.ndarray
```

This type alias makes function signatures more readable: `cosine_similarity(vec_a: Vector, vec_b: Vector)` clearly states the intent.

---

## Self-Test (Lines 466–518)

When run directly (`python core/distance_metrics.py`), validates all functions with test vectors and prints results.
