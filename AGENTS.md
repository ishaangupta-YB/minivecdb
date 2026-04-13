# AGENTS.md — MiniVecDB Project Instructions

## Project Overview

You are building **MiniVecDB**, a mini vector database from scratch in Python for a university Database Management Systems (DBMS) course. The project demonstrates both traditional DBMS concepts (SQL, relational tables, CRUD) AND modern AI-driven database concepts (vector embeddings, similarity search, semantic retrieval).

## Architecture (MUST follow exactly)

**Hybrid storage architecture:**
- **SQLite** (`sqlite3`, built-in) → stores structured data: records table, metadata table (EAV pattern), collections table. All SQL queries use parameterised `?` placeholders. Foreign keys with `ON DELETE CASCADE` for data integrity. `PRAGMA foreign_keys = ON` always.
- **NumPy** (`.npy` files) → stores vector embeddings as a `(N, 384)` float32 matrix. The matrix enables fast batch similarity computation via `matrix @ query_vector`.
- **Bridge** (`id_mapping.json`) → ordered list mapping NumPy row index → record ID in SQLite. `_id_list[i]` is the ID of the vector at `_vectors[i]`.
- **Embedding** → `sentence-transformers` library, model `all-MiniLM-L6-v2`, produces 384-dim float32 vectors. Use `cache_folder` rooted in the project at `db_run/model_cache/huggingface`. Include `SimpleEmbeddingEngine` fallback (bag-of-words) when sentence-transformers unavailable.
- **Search** → built from scratch. Brute-force exact KNN. Three metrics: cosine similarity (default), euclidean distance, dot product. All implemented in `distance_metrics.py` using NumPy. Pre-filter via SQL metadata queries, THEN compute similarity only on filtered candidates.

**Disk layout:**
```
project_root/
└── db_run/
    ├── .active_run                        ← current default run marker
    ├── model_cache/
    │   └── huggingface/                   ← SentenceTransformer cache
    └── demo_<timestamp>_<random>/
        ├── minivecdb.db                   ← SQLite database
        ├── vectors.npy                    ← NumPy array (N, 384) float32
        └── id_mapping.json                ← row index → record ID mapping
```

## Project Structure

```
minivecdb/
├── ARCHITECTURE.py         # Data models, SQL schema, design spec
├── README.md
├── requirements.txt        # numpy, sentence-transformers, flask, pytest
├── core/
│   ├── __init__.py
│   ├── distance_metrics.py # 3 similarity metrics (cosine, euclidean, dot) + batch versions
│   ├── embeddings.py       # EmbeddingEngine + SimpleEmbeddingEngine + factory
│   ├── vector_store.py     # Main VectorStore class (the heart of the project)
│   └── collections.py      # Collection management helpers (optional, can be in vector_store)
├── storage/
│   ├── __init__.py
│   └── database.py         # SQLite wrapper: init schema, execute queries, close
├── cli/
│   └── main.py             # argparse CLI: insert, search, delete, list, stats commands
├── web/
│   ├── app.py              # Flask web app with search UI
│   └── templates/
│       └── index.html      # Search form + results page
├── tests/
│   ├── __init__.py
│   ├── test_distance_metrics.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_database.py
│   └── run_all_tests.py    # Standalone test runner (works without pytest)
├── demo/
│   └── semantic_search.py  # Demo app: load dataset, run searches, show results
└── data/                   # Dataset files for demo
```

## Data Models (defined in ARCHITECTURE.py, import from there)

```python
@dataclass
class VectorRecord:
    id: str                    # Primary key, e.g. "vec_a1b2c3d4"
    vector: np.ndarray         # (384,) float32 — NOT stored in SQLite
    text: str                  # Original text — stored in SQLite
    metadata: Dict[str, Any]   # Key-value tags — stored in SQLite metadata table
    created_at: float          # Unix timestamp
    collection: str            # Which collection, default "default"

@dataclass
class SearchResult:
    record: VectorRecord
    score: float               # Similarity score
    rank: int                  # 1 = best match
    metric: str                # "cosine", "euclidean", or "dot"
```

## SQLite Schema (defined in ARCHITECTURE.py as SCHEMA_SQL)

Three tables: `collections` (name PK), `records` (id PK, FK→collections), `metadata` (record_id FK→records, key, value). Indexes on `records(collection)`, `metadata(key, value)`, `metadata(record_id)`. All queries are in `SQL_QUERIES` dict in ARCHITECTURE.py — always use those templates with parameterised `?` placeholders, never string-format SQL.

## Coding Standards

- **Language:** Python 3.11+, type hints on all function signatures.
- **Comments:** Every function gets a docstring explaining what it does, its parameters, and what it returns. Non-obvious logic gets inline comments. This is a learning project — comments are essential, not optional.
- **Error handling:** Validate inputs, raise `ValueError` with clear messages. Handle SQLite errors with try/except. Never silently fail.
- **Testing:** Every module gets tests. Use `pytest` if available, otherwise provide standalone test runner with `assert`-based checks. Tests must be runnable with `python tests/run_all_tests.py`.
- **Imports:** Import data models from `ARCHITECTURE.py`. Import metrics from `distance_metrics.py`. Import embeddings from `embeddings.py`. Don't redefine these.
- **No external vector DB libraries:** Never use ChromaDB, FAISS, Pinecone, Weaviate, or any vector database library. The vector search is built from scratch. Only allowed libraries: numpy, sentence-transformers, flask, sqlite3 (built-in), pytest, json, os, time, uuid, argparse, dataclasses, typing.
- **Float32:** All vectors stored as `np.float32` to save memory.
- **Auto-save:** After every insert/delete/update, save `vectors.npy` and `id_mapping.json`. SQLite auto-commits.

## Critical Constraints

1. The vector similarity search engine (distance metrics, batch computation, ranking) is 100% built from scratch — this is what makes it a "vector database from scratch."
2. SQLite is used ONLY for structured data storage (records, metadata, collections) — never for storing or searching vectors.
3. NumPy vectors and SQLite records are linked by the `id_mapping.json` bridge file.
4. Always explain code to the user — they are learning as they build. Add teaching comments.
5. When creating files, always include a module docstring explaining the file's purpose.
6. Test everything. Untested code doesn't count.
7. For the CLI, use Python's built-in `argparse`. For the web UI, we will decide in last when our whole DB and CLI is completely ready.
8. The demo must work on a real text dataset (news articles, Wikipedia excerpts, or FAQ entries).