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
| CLI | argparse (built-in) | Command-line interface |
| Web UI | Flask | Browser-based search interface |
| Testing | pytest | Automated test suite |

### Dependencies (requirements.txt)
```
numpy>=1.24.0              # Core math library
sentence-transformers>=2.2.0  # Neural embedding model
flask>=3.0.0               # Web server
pytest>=7.0.0              # Testing framework
```

---

## Project Structure

```
minivecdb/
├── ARCHITECTURE.py          # Central data models + SQL schema + constants
├── AGENTS.md                # AI assistant instructions
├── QUICK_REFERENCE.md       # Student quick-start guide
├── README.md                # Project readme
├── requirements.txt         # Python dependencies
│
├── core/                    # Core engine modules
│   ├── __init__.py          # Package exports
│   ├── distance_metrics.py  # 3 similarity metrics (cosine, euclidean, dot)
│   ├── embeddings.py        # Text → vector conversion engines
│   ├── runtime_paths.py     # Path management for db_run folders
│   └── vector_store.py      # ★ The main VectorStore class (heart of project)
│
├── storage/                 # Database access layer
│   ├── __init__.py          # Package init
│   └── database.py          # SQLite wrapper (Repository pattern)
│
├── cli/                     # Command-line interface
│   ├── __init__.py          # Package init
│   └── main.py              # argparse CLI with 10 subcommands
│
├── web/                     # Web interface (Flask)
│   ├── app.py               # Flask application
│   └── templates/
│       └── index.html       # Search UI template
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_distance_metrics.py
│   ├── test_edge_cases.py
│   ├── test_integration.py
│   ├── test_day5.py ... test_day10.py
│   ├── run_all_tests.py
│   ├── run_tests_distance.py
│   └── run_tests_embeddings.py
│
├── demo/                    # Demo application
│   └── semantic_search.py
│
├── data/                    # Dataset files for demo
│
└── db_run/                  # Runtime artifacts (gitignored)
    ├── .active_run           # Points to current run directory
    ├── model_cache/
    │   └── huggingface/      # Cached embedding model (~80MB)
    └── demo_<timestamp>_<random>/
        ├── minivecdb.db      # SQLite database
        ├── vectors.npy       # NumPy vector matrix (N × 384)
        └── id_mapping.json   # Row index → record ID bridge
```

---

## Key Design Decisions

1. **No external vector DB libraries** — The vector search is built from scratch using NumPy to demonstrate how it actually works.

2. **SQLite for structured data, NumPy for vectors** — SQLite excels at relational queries but can't do fast matrix math. NumPy excels at batch numerical operations but can't do SQL queries. Using both gives us the best of both worlds.

3. **Brute-force exact KNN** — For a learning project, brute-force search (compare query against ALL vectors) is simple to understand. Production databases use approximate methods (ANN) for speed.

4. **Auto-save after every mutation** — Every insert, update, or delete immediately persists to disk. No data loss on crash.

5. **Context manager support** — `with VectorStore(...) as db:` ensures proper cleanup even on exceptions.

6. **Managed run directories** — Each session creates a unique folder (`demo_<timestamp>_<random>`) under `db_run/`, preventing data collisions between experiments.

7. **Embedding engine fallback** — If `sentence-transformers` isn't installed, a simple bag-of-words fallback keeps the rest of the system functional for testing.
