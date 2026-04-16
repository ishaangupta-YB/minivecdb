# MiniVecDB — Project Overview

## What Is MiniVecDB?

MiniVecDB is a **mini vector database built entirely from scratch in Python**. It was created as a university Database Management Systems (DBMS) course project that bridges two worlds:

1. **Traditional DBMS concepts** — SQL, relational tables, CRUD operations, foreign keys, indexes, transactions
2. **Modern AI-driven database concepts** — vector embeddings, semantic similarity search, neural text encoding

The key constraint: **no external vector database libraries are allowed**. No ChromaDB, FAISS, Pinecone, or Weaviate. The entire vector search engine — distance metrics, batch computation, ranking — is built from scratch using only NumPy.

---

## Why Does This Exist?

In a traditional database, you search by **exact match**: `WHERE name = 'Alice'`. But what if you want to find documents that are *similar in meaning* to a query? For example:

- Query: "machine learning algorithms"
- Expected results: documents about neural networks, deep learning, AI — even if they never use the exact words "machine learning algorithms"

This is called **semantic search**, and it works by converting text into high-dimensional vectors (arrays of numbers) that capture meaning. MiniVecDB demonstrates how this works under the hood.

---

## Core Concepts

### 1. Vector Embeddings
A **vector embedding** is a fixed-size array of floating-point numbers that represents the "meaning" of a piece of text. MiniVecDB uses the `all-MiniLM-L6-v2` model from the `sentence-transformers` library, which produces **384-dimensional** vectors.

- *"The cat sat on the mat"* → `[0.12, -0.03, 0.45, ...]` (384 numbers)
- *"A kitten rested on a rug"* → `[0.11, -0.02, 0.44, ...]` (similar numbers!)
- *"Stock market crashed today"* → `[-0.55, 0.21, -0.33, ...]` (very different numbers)

Texts with similar meanings produce vectors that "point in the same direction" in 384-dimensional space.

### 2. Similarity Search
Once text is converted to vectors, finding similar documents becomes a math problem: **compute the distance/angle between vectors**. MiniVecDB supports three metrics:

| Metric | Formula | Range | Best Score | Use Case |
|--------|---------|-------|------------|----------|
| Cosine Similarity | A·B / (‖A‖·‖B‖) | [-1, 1] | 1.0 | Text (direction-based, ignores length) |
| Euclidean Distance | √(Σ(Aᵢ-Bᵢ)²) | [0, ∞) | 0.0 | Spatial data (straight-line distance) |
| Dot Product | Σ(Aᵢ×Bᵢ) | (-∞, ∞) | Higher | Speed (with normalized vectors) |

### 3. Hybrid Storage
MiniVecDB uses a **hybrid storage architecture** — two different storage engines for two different types of data:

| Data Type | Stored In | Why |
|-----------|-----------|-----|
| Structured data (text, metadata, collections) | **SQLite** | SQL queries, ACID transactions, foreign keys |
| Vector embeddings (arrays of 384 floats) | **NumPy .npy files** | Fast batch math, compact binary format |
| Bridge (row index ↔ record ID) | **JSON file** | Links the two systems together |

### 4. Pre-Filtering
Before computing expensive vector similarity, MiniVecDB can narrow candidates using SQL metadata queries. For example:
```
"Find documents about AI published after 2020"
→ Step 1: SQL query filters metadata for year > 2020
→ Step 2: Vector search only on those filtered candidates
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.11+ | Core language |
| Structured Storage | sqlite3 (built-in) | Records, metadata, collections |
| Vector Storage | NumPy | Embedding matrices, batch math |
| Embedding Model | sentence-transformers | Text → 384-dim vector conversion |
| File Processing | csv + pandas + openpyxl | Robust TXT/CSV/Excel ingestion with tabular header normalization |
| CLI | argparse (built-in) | Command-line interface |
| Web UI | Flask | Browser-based search interface |
| Testing | pytest + run_all_tests.py | Automated suite (run with `.venv/bin/python -m pytest tests/ -v` or `.venv/bin/python tests/run_all_tests.py`) |

### Dependencies (requirements.txt)
```
numpy>=1.24.0              # Core math library
sentence-transformers>=2.2.0  # Neural embedding model
flask>=3.0.0               # Web server
pandas>=2.0.0              # CSV/Excel file parsing
openpyxl>=3.1.0            # Excel (.xlsx) support for pandas
pytest>=7.0.0              # Testing framework
```

---

## Project Structure

```
minivecdb/
├── ARCHITECTURE.py          # Central data models + SQL schema (v3.0: 6 tables + 3 triggers)
├── AGENTS.md                # AI assistant instructions
├── QUICK_REFERENCE.md       # Student quick-start guide
├── README.md                # Project readme
├── requirements.txt         # Python dependencies
│
├── core/                    # Core engine modules
│   ├── __init__.py          # Package exports
│   ├── distance_metrics.py  # 3 similarity metrics (cosine, euclidean, dot)
│   ├── embeddings.py        # Text → vector conversion engines
│   ├── file_processor.py    # File upload pipeline: validate → extract/normalize → format-specific chunk
│   ├── runtime_paths.py     # Path management for db_run folders
│   └── vector_store.py      # ★ The main VectorStore class (heart of project)
│
├── storage/                 # Database access layer
│   ├── __init__.py          # Package init
│   ├── database.py          # SQLite wrapper (session-bound, v3.0)
│   └── migrations.py        # Legacy per-session DB migration to shared DB
│
├── cli/                     # Command-line interface
│   ├── __init__.py          # Package init
│   └── main.py              # argparse CLI with 10 subcommands
│
├── web/                     # Web interface (Flask, v3.0)
│   ├── __init__.py          # Package init
│   ├── app.py               # Flask application (10 routes + JSON API)
│   └── templates/
│       ├── _base.html       # Shared layout + CSS design system
│       ├── select_session.html  # Session picker landing page
│       ├── index.html       # Search form
│       ├── results.html     # Search results
│       ├── insert.html      # Insert form
│       ├── stats.html       # Database statistics
│       └── history.html     # Chat history timeline
│
├── tests/                   # Test suite + benchmarks
│   ├── __init__.py
│   ├── test_distance_metrics.py
│   ├── test_sessions_schema.py  # v3.0 triggers, cascade, aggregates
│   ├── test_edge_cases.py
│   ├── test_integration.py
│   ├── test_day5.py ... test_day10.py
│   ├── benchmark.py         # Performance benchmark suite
│   ├── benchmark_results.json  # Saved benchmark results
│   ├── run_all_tests.py
│   ├── run_tests_distance.py
│   └── run_tests_embeddings.py
│
├── demo/                    # Demo application
│   └── semantic_search.py   # 6-step end-to-end demo
│
├── data/                    # Curated dataset (150+ docs)
│   ├── sample_dataset.py    # Dataset loader
│   └── generated/           # 5 JSON shard files
│
└── db_run/                  # Runtime artifacts (gitignored)
    ├── .active_run           # Points to current session folder
    ├── minivecdb.db          # SHARED SQLite DB (all sessions)
    ├── model_cache/
    │   └── huggingface/      # Cached embedding model (~80MB)
    └── demo_<timestamp>_<random>/
        ├── vectors.npy       # NumPy vector matrix (N × 384)
        └── id_mapping.json   # Row index → record ID bridge
```

---

## Key Design Decisions

1. **No external vector DB libraries** — The vector search is built from scratch using NumPy to demonstrate how it actually works.

2. **SQLite for structured data, NumPy for vectors** — SQLite excels at relational queries but can't do fast matrix math. NumPy excels at batch numerical operations but can't do SQL queries. Using both gives us the best of both worlds.

3. **Shared DB (v3.0)** — All sessions share a single `db_run/minivecdb.db` file. Sessions are isolated at the query layer via `session_id` foreign keys. Only vector artifacts (`vectors.npy`, `id_mapping.json`) are per-session on disk.

4. **Brute-force exact KNN** — For a learning project, brute-force search (compare query against ALL vectors) is simple to understand. Production databases use approximate methods (ANN) for speed.

5. **Auto-save after every mutation** — Every insert, update, or delete immediately persists to disk. No data loss on crash.

6. **Context manager support** — `with VectorStore(...) as db:` ensures proper cleanup even on exceptions.

7. **Managed run directories** — Each session creates a unique folder (`demo_<timestamp>_<random>`) under `db_run/`, preventing data collisions between experiments.

8. **Embedding engine fallback** — If `sentence-transformers` isn't installed, a simple bag-of-words fallback keeps the rest of the system functional for testing.

9. **Legacy migration** — Pre-v3 per-session DBs are automatically migrated to the shared DB. The old file is renamed to `.legacy` — user data is never deleted.
