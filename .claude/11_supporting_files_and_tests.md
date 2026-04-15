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

### `web/__init__.py`
**Lines**: 1 | **Size**: 39 bytes

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

The project has **16 test files** organized by feature area and development day:

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
| `test_sessions_schema.py` | 6 tests | v3.0: triggers, cascade deletes, aggregate queries, session binding |
| `test_integration.py` | ~10 tests | Cross-module integration tests |
| `test_edge_cases.py` | ~10 tests | Boundary conditions and error cases |
| `day2_3_integration_test.py` | ~5 tests | Early integration (Days 2-3) |
| `benchmark.py` | 4 benchmarks | Insertion throughput, query latency, memory, metric comparison |
| `benchmark_results.json` | — | Saved JSON results from last benchmark run |
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
db_run/                 # All runtime artifacts (databases, vectors, model cache)
__pycache__/            # Python bytecode
*.py[cod]               # Compiled Python
.venv/                  # Virtual environment
.pytest_cache/          # Pytest cache
.claude/                # Documentation (generated)
benchmark_results.json  # Generated benchmark output
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
├── minivecdb.db                         # SHARED SQLite DB (all sessions)
├── model_cache/
│   └── huggingface/
│       └── sentence-transformers_all-MiniLM-L6-v2/  # ~80MB cached model
│
├── demo_1713052800_a1b2c3/              # Session 1 (active)
│   ├── vectors.npy                      # NumPy binary (N × 384 × 4 bytes)
│   └── id_mapping.json                  # JSON list (row index → record ID)
│
└── demo_1713139200_d4e5f6/              # Session 2 (inactive)
    ├── vectors.npy
    └── id_mapping.json
```

**Note (v3.0)**: Only `vectors.npy` and `id_mapping.json` are per-session on disk. All tabular data lives in the shared `minivecdb.db` at the `db_run/` root, with sessions isolated at the query layer via `session_id` foreign keys.

---

## Web Module ✅

### `web/` Directory

Now fully implemented with:
- `web/__init__.py` — Package marker
- `web/app.py` (510 lines, 16.9 KB) — Flask application with 10 routes: session picker, search, insert, stats, history, JSON API
- `web/templates/` — 7 Jinja2 templates with a complete CSS design system

> See [13_file_web_app.md](./13_file_web_app.md) for the complete breakdown.

---

## Demo Module ✅

### `demo/` Directory

Now populated with:
- `demo/__init__.py` — Package marker
- `demo/semantic_search.py` (353 lines, 14.9 KB) — Full end-to-end demo: loads 150+ docs, inserts, runs 10 queries, demonstrates filtered search, semantic similarity comparisons

> See [12_file_data_benchmarks_demo.md](./12_file_data_benchmarks_demo.md) for a complete breakdown.

---

## Data Module ✅

### `data/` Directory

Now populated with a curated dataset of **150+ documents** across 5 categories (Technology, Science, Sports, Health, Business), split into 5 JSON shard files under `data/generated/`. Includes a loader module (`sample_dataset.py`) and used by both the demo and benchmarks.

> See [12_file_data_benchmarks_demo.md](./12_file_data_benchmarks_demo.md) for a complete breakdown.
