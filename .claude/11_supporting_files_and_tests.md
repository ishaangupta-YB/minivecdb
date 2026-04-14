# MiniVecDB — Supporting Files & Tests

> This document covers the remaining files: package inits, tests, and configuration.

---

## Package Init Files

### `core/__init__.py`
**Lines**: 40 | **Size**: 1.2 KB

**What it does**: Exports all public symbols from the `core` package so users can write:
```python
from core import cosine_similarity, EmbeddingEngine
```

**Key behavior**:
- Always exports distance metric functions (`cosine_similarity`, `batch_cosine_similarity`, etc.)
- Conditionally exports embedding classes only if the `embeddings` module loads successfully
- Uses `try/except` around `from . import embeddings` to handle missing dependencies gracefully

### `storage/__init__.py`
**Lines**: 1 | **Size**: 59 bytes

Just a docstring. Marks the directory as a Python package.

### `cli/__init__.py`
**Lines**: 1 | **Size**: 28 bytes

Empty package marker.

### `tests/__init__.py`
**Lines**: 1 | **Size**: 34 bytes

Empty package marker.

---

## Requirements File

### `requirements.txt`
**Lines**: 15 | **Size**: 274 bytes

```
numpy>=1.24.0              # Core math (required)
sentence-transformers>=2.2.0  # Neural embeddings (required for quality)
flask>=3.0.0               # Web UI (optional)
pytest>=7.0.0              # Test framework (development)
```

**Note**: The project gracefully degrades without `sentence-transformers` (falls back to SimpleEmbeddingEngine) and without `flask` (web UI just won't work). Only `numpy` is truly mandatory.

---

## Test Files

### Test Organization

The project has **13 test files** organized by feature area and development day:

| File | Tests | Focus Area |
|------|-------|-----------|
| `test_distance_metrics.py` | 46 tests | All 3 metrics (single + batch), edge cases |
| `run_tests_distance.py` | 46 tests | Standalone runner (works without pytest) |
| `run_tests_embeddings.py` | 17 tests | Embedding engine validation |
| `test_day5.py` | ~20 tests | Insert + Get (Day 5 features) |
| `test_day6.py` | ~15 tests | Search engine (Day 6 features) |
| `test_day7.py` | ~15 tests | Delete + Update + Collections (Day 7) |
| `test_day8.py` | ~15 tests | Persistence save/load (Day 8) |
| `test_day10.py` | ~15 tests | Advanced metadata filtering (Day 10) |
| `test_integration.py` | ~10 tests | Cross-module integration tests |
| `test_edge_cases.py` | ~10 tests | Boundary conditions and error cases |
| `day2_3_integration_test.py` | ~5 tests | Early integration (Days 2-3) |
| `run_all_tests.py` | — | Discovers and runs all tests |

### Running Tests

```bash
# With pytest (recommended)
python -m pytest tests/ -v

# Without pytest (standalone runner)
python tests/run_all_tests.py

# Individual test files
python -m pytest tests/test_distance_metrics.py -v
python tests/run_tests_distance.py  # standalone
```

### `run_all_tests.py`
**Lines**: ~100 | A standalone test runner that works without pytest.

Uses `importlib` to discover and import all `test_*.py` and `run_tests_*.py` files, then runs their test functions using `assert`-based checks.

### Key Test Patterns

```python
# Distance metrics test example
def test_identical_vectors_cosine():
    """Identical vectors should have cosine similarity of 1.0"""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

# VectorStore integration test example
def test_insert_and_search():
    """Insert a record and find it via search"""
    with VectorStore(storage_path=tmpdir) as store:
        rid = store.insert("Python programming")
        results = store.search("coding in Python", top_k=1)
        assert results[0].record.id == rid
```

---

## Configuration Files

### `.gitignore`
**Size**: 379 bytes

Key entries:
```
db_run/           # All runtime artifacts (databases, vectors, model cache)
__pycache__/      # Python bytecode
*.pyc
.venv/            # Virtual environment
.pytest_cache/    # Pytest cache
```

**Why `db_run/` is gitignored**: This directory contains:
- SQLite databases (large, binary, user-specific)
- NumPy vector files (large, binary)
- Model cache (80MB+ of downloaded weights)

None of these should be version-controlled.

---

## Disk Layout (Runtime)

When MiniVecDB is used, the `db_run/` directory grows:

```
db_run/
├── .active_run                          # Text file: "demo_1713052800_a1b2c3"
├── model_cache/
│   └── huggingface/
│       └── sentence-transformers_all-MiniLM-L6-v2/  # ~80MB cached model
│
├── demo_1713052800_a1b2c3/              # Run 1 (active)
│   ├── minivecdb.db                     # SQLite database (~100KB per 1000 records)
│   ├── vectors.npy                      # NumPy binary (1000 records × 384 × 4 = ~1.5MB)
│   └── id_mapping.json                  # JSON list (~40KB per 1000 records)
│
└── demo_1713139200_d4e5f6/              # Run 2 (old, inactive)
    ├── minivecdb.db
    ├── vectors.npy
    └── id_mapping.json
```

---

## Web Module (Placeholder)

### `web/` Directory

Currently empty (no `app.py` or templates). The AGENTS.md specifies this will be built as a Flask web interface in Day 15, with:
- `web/app.py` — Flask application with search API endpoints
- `web/templates/index.html` — Search form + results page

The architecture supports this: `app.py` would create a `VectorStore` instance and expose `store.search()` through HTTP endpoints.

---

## Demo Module (Placeholder)

### `demo/` Directory

Currently empty. Planned for Day 14:
- `demo/semantic_search.py` — Loads a real text dataset, runs example searches, displays results
- Would demonstrate the full stack: insert documents → search → see ranked results

---

## Data Module (Placeholder)

### `data/` Directory

Currently empty. Will hold dataset files for the demo (news articles, Wikipedia excerpts, or FAQ entries).
