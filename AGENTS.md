# AGENTS.md — MiniVecDB Project Instructions

## Project Overview

You are building **MiniVecDB**, a mini vector database from scratch in Python for a university Database Management Systems (DBMS) course. The project demonstrates both traditional DBMS concepts (SQL, relational tables, CRUD, JOINs, subqueries, aggregates, triggers, 3NF normalization) AND modern AI-driven database concepts (vector embeddings, similarity search, semantic retrieval).

## Architecture (MUST follow exactly)

**Hybrid storage architecture:**
- **SQLite** (`sqlite3`, built-in) → stores all structured data in a single **shared** database: sessions, conversations, messages (chat history), collections, records, metadata (EAV). All SQL queries use parameterised `?` placeholders. Foreign keys with `ON DELETE CASCADE` for data integrity. `PRAGMA foreign_keys = ON` always. Schema is 3NF-normalized.
- **NumPy** (`.npy` files, **per-session**) → stores vector embeddings as a `(N, 384)` float32 matrix. The matrix enables fast batch similarity computation via `matrix @ query_vector`.
- **Bridge** (`id_mapping.json`, **per-session**) → ordered list mapping NumPy row index → record ID in SQLite. `_id_list[i]` is the ID of the vector at `_vectors[i]`.
- **Embedding** → `sentence-transformers` library, model `all-MiniLM-L6-v2`, produces 384-dim float32 vectors. Uses `cache_folder` rooted at `db_run/model_cache/huggingface`. Includes `SimpleEmbeddingEngine` fallback (bag-of-words) when sentence-transformers unavailable.
- **Runtime path manager** → `core/runtime_paths.py` controls project-root path resolution, active run selection, unique run naming (`demo_<timestamp>_<random>`), cache path creation, shared DB path resolution, and run folder enumeration.
- **Search** → built from scratch. Brute-force exact KNN. Three metrics: cosine similarity (default), euclidean distance, dot product. All implemented in `distance_metrics.py` using NumPy. Pre-filter via SQL metadata queries (session-scoped), THEN compute similarity only on filtered candidates.
- **Legacy migration** → `storage/migrations.py` detects pre-v3 per-session `minivecdb.db` files, copies their collections/records/metadata into the shared DB under a new session row, and renames the old file to `minivecdb.db.legacy` (never deletes user data).

**Disk layout (v3.0):**
```
project_root/
└── db_run/
    ├── .active_run                        ← folder name of currently-active session
    ├── minivecdb.db                       ← SHARED SQLite DB (all sessions/conversations/messages/records)
    ├── model_cache/
    │   └── huggingface/                   ← SentenceTransformer cache
    └── demo_<timestamp>_<random>/         ← one folder per session
        ├── vectors.npy                    ← NumPy array (N, 384) float32
        └── id_mapping.json                ← row index → record ID mapping
```

Only `vectors.npy` and `id_mapping.json` are per-session on disk. All tabular data lives in one shared DB; sessions are isolated at the query layer via `session_id` foreign keys.

Default behavior:
- **No auto-seed.** The web app shows a session picker on first visit — the user either starts a new session or resumes an existing one. No sample dataset is loaded automatically.
- CLI takes `--session <folder_name>` to bind a specific session, or `--new-run` to create one. If neither is passed and `.active_run` is empty, the CLI lists available sessions and exits.
- `--new-run` creates a fresh unique run folder, registers it in `sessions`, and updates `.active_run`.

## Project Structure

```
minivecdb/
├── ARCHITECTURE.py         # Data models, SQL schema (v3.0: 6 tables + 3 triggers), SQL_QUERIES
├── AGENTS.md
├── README.md
├── requirements.txt        # numpy, sentence-transformers, flask, pytest
├── core/
│   ├── __init__.py
│   ├── distance_metrics.py # 3 similarity metrics (cosine, euclidean, dot) + batch versions
│   ├── embeddings.py       # EmbeddingEngine + SimpleEmbeddingEngine + factory
│   ├── runtime_paths.py    # Run folders, active run marker, cache paths, shared DB path, run listing
│   └── vector_store.py     # Main VectorStore class (session-aware, shared-DB routing)
├── storage/
│   ├── __init__.py
│   ├── database.py         # Session-bound DatabaseManager: record/metadata/collection CRUD + log_message/get_history
│   └── migrations.py       # One-shot migration of legacy per-session DBs into the shared DB
├── cli/
│   └── main.py             # argparse CLI: insert/search/delete/list/stats + --session/--new-run
├── web/
│   ├── app.py              # Flask app: session picker + search/insert/history/stats + /api/search
│   └── templates/
│       ├── _base.html          # Shared layout: header, nav (gated on active session), session banner
│       ├── select_session.html # Landing page: new session + resume dropdown + sessions table
│       ├── index.html          # Search form (session-scoped)
│       ├── results.html        # Search results page
│       ├── insert.html         # Insert form
│       ├── stats.html          # Per-session stats
│       └── history.html        # Chat history timeline (messages table for active session)
├── tests/
│   ├── __init__.py
│   ├── test_distance_metrics.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_database.py
│   ├── test_sessions_schema.py # v3.0: triggers, cascade deletes, aggregate queries, session binding
│   └── run_all_tests.py        # Standalone test runner (works without pytest)
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
    collection: str            # Collection name (scoped to the session), default "default"

@dataclass
class SearchResult:
    record: VectorRecord
    score: float               # Similarity score
    rank: int                  # 1 = best match
    metric: str                # "cosine", "euclidean", or "dot"

@dataclass
class DatabaseStats:
    total_records: int
    total_collections: int
    collections: List[Tuple[str, int]]
    db_size_bytes: int
    vector_count: int
    session_name: str          # v3.0: which session these stats belong to

@dataclass
class SessionInfo:
    id: int
    name: str
    storage_path: str
    created_at: float
    last_used_at: float
    msg_count: int
    record_count: int

@dataclass
class MessageRow:
    id: int
    created_at: float
    kind: str                  # 'search' | 'insert'
    query_text: str
    metric: Optional[str]
    top_k: Optional[int]
    category_filter: Optional[str]
    result_count: Optional[int]
    elapsed_ms: Optional[float]
    response_ref: Optional[str]
```

## SQLite Schema (v3.0 — defined in ARCHITECTURE.py as SCHEMA_SQL)

**Six tables, 3NF-normalized, with three triggers:**

1. **`sessions`** — one row per session folder. `name UNIQUE`, `storage_path`, `created_at`, `last_used_at`.
2. **`conversations`** — one-to-many with sessions. Every session gets a "Default conversation" auto-created by trigger.
3. **`messages`** — chat history of **user queries only** (no result rows). Columns: `kind` ('search'|'insert'), `query_text`, `metric`, `top_k`, `category_filter`, `result_count`, `elapsed_ms`, `response_ref`. FK → conversations.
4. **`collections`** — session-scoped, `UNIQUE(session_id, name)`. Surrogate INTEGER PK. Every session gets a `default` collection auto-created by trigger.
5. **`records`** — `id TEXT PK` (still `vec_xxxx`), FKs to both `sessions` and `collections`.
6. **`metadata`** — EAV: `(record_id, key, value)`. FK → records.

**Triggers:**
- `trg_create_default_conversation` — AFTER INSERT ON sessions → inserts default conversation.
- `trg_create_default_collection` — AFTER INSERT ON sessions → inserts default collection.
- `trg_touch_session_on_message` — AFTER INSERT ON messages → uses a **subquery** to hop `conversation_id → session_id` and bump `sessions.last_used_at`.

**Where each SQL technique is demonstrated:**
- **JOIN (LEFT JOIN)** — `list_sessions_with_counts`: joins sessions → conversations → messages for the picker.
- **JOIN (INNER JOIN)** — `history_for_session`: joins messages → conversations; `filter_by_metadata_in_session`: joins metadata → records for session-scoped filtering.
- **Subquery** — `trg_touch_session_on_message` resolves `session_id` via `(SELECT session_id FROM conversations WHERE id = NEW.conversation_id)`.
- **Aggregate / GROUP BY** — `list_sessions_with_counts` (msg_count, record_count per session); `count_records_in_session`.
- **Triggers** — three, listed above.

All queries live in `SQL_QUERIES` in `ARCHITECTURE.py`. Always use those templates with parameterised `?` placeholders — never string-format SQL.

## DatabaseManager contract

`DatabaseManager(db_path, session_name=None, session_storage_path=None)` is **session-bound**: every record / metadata / collection method is implicitly scoped to `self.session_id`. On construction it:
1. Opens the shared DB at `db_path`, executes `SCHEMA_SQL`, enables FKs.
2. Upserts the session row (triggers auto-create default conversation + default collection).
3. Caches `self.session_id` and `self.default_conversation_id`.

New v3.0 methods:
- `register_session(name, storage_path) -> int`
- `list_sessions_with_counts() -> List[SessionInfo]`
- `log_message(kind, query_text, *, metric, top_k, category_filter, result_count, elapsed_ms, response_ref) -> int`
- `get_history(limit=200) -> List[MessageRow]`

## VectorStore contract

`VectorStore(storage_path=None, session_name=None, new_run=False, ...)`:
- If `new_run`, creates a fresh folder via `create_new_run_path`, registers it in `sessions`, updates `.active_run`.
- If `session_name` given, binds to that session (upsert).
- Else reads `.active_run`; if still empty, raises `ValueError` (no more silent auto-construction of a run — breaking change in v3.0).
- Routes the DB path through `is_within_db_run(storage_path)`: shared DB when inside `db_run/`, per-folder DB otherwise (keeps legacy test fixtures working).

## Web app contract (Flask)

Routes:
- `GET /` — session picker (`select_session.html`).
- `POST /session/new` — create run, rebind VectorStore, redirect to search page.
- `POST /session/switch` — rebind VectorStore to chosen session, redirect.
- `GET /search-page` — search form; redirects to `/` if no active session.
- `POST /search` — runs query, logs a `messages` row with `kind='search'`. Accepts `filter_key` + `filter_value` for arbitrary metadata pre-filtering (e.g. `category:Science`, `source:sample`, `book:harry`).
- `POST /insert` — adds record, logs a `messages` row with `kind='insert'` and `response_ref=<new_id>`.
- `GET /history` — renders message timeline from `db.get_history()`.
- `GET /stats` — per-session aggregate stats.
- `GET /api/search` — JSON endpoint, same logging as `/search`. Accepts `filter_key` + `filter_value` params (also supports legacy `category` param for backwards compat).

## Coding Standards

- **Language:** Python 3.11+, type hints on all function signatures.
- **Comments:** Every function gets a docstring explaining what it does, its parameters, and what it returns. Non-obvious logic gets inline comments. This is a learning project — comments are essential, not optional.
- **Error handling:** Validate inputs, raise `ValueError` with clear messages. Handle SQLite errors with try/except. Never silently fail.
- **Testing:** Every module gets tests. Use `pytest` if available, otherwise provide standalone test runner with `assert`-based checks. Tests must be runnable with `python tests/run_all_tests.py`.
- **Imports:** Import data models from `ARCHITECTURE.py`. Import metrics from `distance_metrics.py`. Import embeddings from `embeddings.py`. Don't redefine these.
- **No external vector DB libraries:** Never use ChromaDB, FAISS, Pinecone, Weaviate, or any vector database library. The vector search is built from scratch. Only allowed libraries: numpy, sentence-transformers, flask, sqlite3 (built-in), pytest, json, os, time, uuid, argparse, dataclasses, typing.
- **Float32:** All vectors stored as `np.float32` to save memory.
- **Auto-save:** After every insert/delete/update, save `vectors.npy` and `id_mapping.json`. SQLite auto-commits.
- **Runtime artifacts:** Keep all generated runtime files inside `db_run/` and ensure `db_run/` stays in `.gitignore`.
- **Never silently delete user data:** Legacy DB migration renames to `.legacy`, never unlinks.

## Critical Constraints

1. The vector similarity search engine (distance metrics, batch computation, ranking) is 100% built from scratch — this is what makes it a "vector database from scratch."
2. SQLite is used ONLY for structured data storage (sessions, conversations, messages, collections, records, metadata) — never for storing or searching vectors.
3. NumPy vectors and SQLite records are linked by the `id_mapping.json` bridge file.
4. Always explain code to the user — they are learning as they build. Add teaching comments.
5. When creating files, always include a module docstring explaining the file's purpose.
6. Test everything. Untested code doesn't count.
7. For the CLI, use Python's built-in `argparse`. The web UI uses Flask + Jinja2 templates.
8. The demo must work on a real text dataset (news articles, Wikipedia excerpts, or FAQ entries).
9. All tabular state lives in the shared DB at `db_run/minivecdb.db`. Only vector artefacts are per-session on disk.
10. Chat history stores **user queries only** — never per-result rows or large result blobs. Response metadata (`result_count`, `elapsed_ms`, `response_ref`) sits on the same message row.
