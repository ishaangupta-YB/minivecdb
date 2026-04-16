# MiniVecDB — Project Report

## List of Figures *(page i)*

| Fig. | Title | Chapter / Page |
|------|-------|----------------|
| 1 | Layered Architecture | Chapter 1 and Chapter 5 |
| 2 | MiniVecDB Entity-Relationship Diagram (v3.0) | Chapter 4, Section 4.2 |
| 3 | INSERT Data Flow | Chapter 5 |
| 4 | SEARCH Data Flow | Chapter 5 |
| 5 | Three-Way Bridge (SQLite / NumPy / JSON) | Chapter 5 |

---

## List of Tables *(page ii)*

| Table | Title | Chapter / Page |
|-------|-------|----------------|
| 1 | Technology Stack | Chapter 1 |
| 2 | Distance Metrics Comparison | Chapter 5, Section 5.4 |
| 3 | Six-Table Schema Summary | Chapter 4, Section 4.3 |
| 4 | SQL Techniques Demonstrated | Chapter 6, Section 6.4 |
| 5 | Benchmark Summary | Chapter 6, Section 6.2 |

---

## Abstract *(page iii)*

MiniVecDB is a hybrid-storage vector database implemented in Python for a university Database Management Systems (DBMS) course. Structured state — sessions, conversations, messages, collections, records, and entity–attribute–value metadata — lives in a single shared SQLite database at `db_run/minivecdb.db`, while each session persists a `(N, 384)` float32 embedding matrix in NumPy (`vectors.npy`) and an ordered row-to-ID list in `id_mapping.json`, forming a bridge between relational rows and matrix rows. Text is encoded with `sentence-transformers` using the `all-MiniLM-L6-v2` model (384 dimensions); search is exact brute-force *k*-nearest neighbours over filtered candidates, with cosine similarity (default), euclidean distance, and dot product implemented from scratch in NumPy — without FAISS, Pinecone, ChromaDB, Weaviate, or similar vector-database libraries. The SQLite layer is a six-table, third-normal-form design with three triggers for default conversation and collection creation and for session activity timestamps. A command-line interface (`argparse`) and a Flask web application with Jinja2 templates provide insertion, semantic search, statistics, and a session picker; user queries for search and insert are logged as chat-style history in the `messages` table (user queries only, no large per-result blobs). Bulk ingestion supports TXT, CSV, and Excel via shared file-processing logic. Legacy pre-v3.0 storage folders (`minivecdb_data/`, `vectorstore_data/`) are imported into `db_run/` by `VectorStore._maybe_migrate_legacy_storage` and the originals are left on disk, so user data is never silently deleted.

---

## Chapter 1: Introduction *(page 4)*

Relational database systems excel at *exact* retrieval: a row matches when a predicate compares equal to stored values. Much real-world information retrieval is not exact. Users ask in natural language and expect documents that *mean* the same thing even when wording differs. For example, a query such as "machine learning algorithms" should surface text about neural networks, deep learning, and artificial intelligence even when that exact phrase never appears. That shift — from symbolic equality to statistical similarity — is the same broad trajectory that took machine learning from classic algorithms toward neural networks, deep learning, and modern AI: representations become dense vectors learned from data, and "nearness" in vector space proxies semantic nearness in language.

**MiniVecDB** is a compact vector database built entirely from scratch in Python for a DBMS course. It is deliberately *dual-purpose*: on one side it is a classical database project with a normalized relational schema, CRUD paths, joins, subqueries, aggregates, and triggers; on the other it is a contemporary AI-flavoured system that embeds text, stores vectors, and ranks by similarity. A hard course rule forbids outsourcing the vector engine: implementations must not use ChromaDB, FAISS, Pinecone, Weaviate, or any external vector-database library. Permitted dependencies include NumPy, `sentence-transformers`, Flask, pandas, `openpyxl`, the built-in `sqlite3` module, and `pytest`, plus the Python standard library.

At the architectural level (see **Figure 1**, Layered Architecture, in Chapter 5), the system separates concerns: SQLite holds all tabular data for every session in one shared file; each session folder under `db_run/` holds only `vectors.npy` and `id_mapping.json`, linking NumPy row indices to SQLite record identifiers. **Table 1** summarises the technology stack.

**Table 1 — Technology Stack**

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11+ | Core implementation and typing |
| Structured Storage | sqlite3 (built-in) | Shared relational DB, sessions, CRUD, SQL |
| Vector Storage | NumPy | `(N, 384)` float32 matrices, batch similarity |
| Embedding Model | sentence-transformers `all-MiniLM-L6-v2` | Text to 384-d embeddings (with bag-of-words fallback) |
| File Processing | csv + pandas + openpyxl | TXT / CSV / Excel ingest and normalisation |
| CLI | argparse | Subcommands: `insert`, `search`, `get`, `update`, `delete`, `list`, `stats`, `collections`, `create-collection`, `import-file` |
| Web UI | Flask + Jinja2 | Session picker, HTML search + `/api/search` JSON endpoint, insert, file upload, records browser, history, stats |
| Testing | pytest + `tests/run_all_tests.py` | Automated tests and standalone runner |
| Benchmarking | NumPy | Performance experiments aligned with batch vector math |

Together, these choices make MiniVecDB a readable end-to-end artefact: SQL demonstrates relational discipline and session isolation, while NumPy makes the similarity computation inspectable line by line rather than hidden behind a proprietary index.

---

## Chapter 2: Related Work *(page 5)*

Vector data management today spans specialised libraries, managed cloud services, and hybrid search engines. **ChromaDB** is a popular open-source embedding-oriented store: it emphasises collections of documents, metadata, and persistence for retrieval applications. MiniVecDB borrows the *idea* of session- or project-scoped collections as named groupings of records, but keeps persistence in SQLite and vectors on disk as NumPy arrays rather than adopting Chroma's stack.

**FAISS** (Facebook AI Similarity Search) is a high-performance library for similarity search over dense vectors. It provides approximate nearest neighbour (ANN) structures — IVF, HNSW, product quantisation, and related schemes — to avoid comparing a query against every vector when collections grow large. MiniVecDB intentionally does **not** use FAISS: the course objective is pedagogical transparency, so search is brute-force exact *k*-NN after optional SQL pre-filtering, with scores from explicit batch matrix–vector operations in NumPy.

**Pinecone** is a managed vector database typically used in deployed ML systems: hosting, scaling, and operations are abstracted away from the application. For a local, inspectable DBMS assignment, a from-scratch design that runs entirely on a laptop — shared SQLite file, per-session vector files, no cloud dependency — matches educational goals better than coupling the project to a hosted service.

**Weaviate** and similar systems often blend vector search with structured and graph-flavoured features (classes, schemas, hybrid retrieval). MiniVecDB's hybrid aspect is narrower: relational metadata and chat history in SQLite, vectors outside SQLite, joined at the application layer via the bridge file.

**SQLite FTS5** provides powerful *keyword* full-text search inside the database engine. It ranks and matches on tokens and linguistic rules, not on learned semantic embeddings. MiniVecDB uses embeddings precisely where FTS stops: to align meaning across different surface forms, after optionally narrowing candidates with SQL on metadata.

**Classic RDBMS** products such as MySQL or PostgreSQL remain the reference for transactional SQL and exact predicates. They establish the baseline — schemas, keys, and query expressiveness — that MiniVecDB follows for structured data, while delegating similarity to NumPy.

In sum, no single component of MiniVecDB is novel in isolation: normalised schemas, triggers, and brute-force vector math are all well-known building blocks. The contribution for this project is *synthetic* and *didactic*: implement the vector-search engine from first principles using NumPy batch mathematics, wire it cleanly to a traditional relational model in SQLite, and expose the behaviour through CLI and web interfaces — without treating retrieval as an opaque library call.

---

## Chapter 3: Problem Statement and Objectives *(page 7)*

**Problem Statement.** A DBMS capstone should exercise both classical relational skills and modern retrieval that students actually encounter in AI-backed applications. Course-friendly vector stacks often lean on high-level libraries that conceal indexing, distance computation, and ranking. That speeds prototypes but obscures the database-adjacent mechanics this course is meant to teach: how to normalise data, how SQL constrains and connects entities, and how application code stitches heterogeneous stores together.

MiniVecDB is framed around a realistic usage pattern: a user works inside a session, loads or pastes text, runs semantic search over stored documents, and sees history of their own queries — not dumps of full result payloads — captured for reproducibility and auditing. Session isolation must hold in SQL (foreign keys and scoped queries) while vectors remain efficient to scan in bulk for similarity. The problem is therefore to unify a rigorous relational substrate with an explicit, inspectable vector pipeline, without relying on black-box vector-database products.

Bridging these worlds requires clear interfaces: parameterised SQL for integrity and filtering, a NumPy matrix for embeddings, and a small JSON mapping to connect row order to primary keys. Recovery and migration must respect user data: if older per-session databases exist, they should be absorbed into the shared database and the legacy files renamed rather than deleted.

**Objectives.**

1. Design a third-normal-form SQLite schema with six tables: `sessions`, `conversations`, `messages`, `collections`, `records`, and `metadata`, with foreign keys and cascade deletes as described in project architecture.
2. Implement CRUD operations for records and metadata (and related collection/session operations) through a session-bound database access layer.
3. Demonstrate SQL techniques including JOINs (inner and left), subqueries, aggregates with `GROUP BY`, triggers, `CHECK` constraints (for example on `messages.kind`), composite `UNIQUE` constraints on collections, `UPSERT`/`ON CONFLICT` for idempotent session registration, and cascade deletes across the relationship graph.
4. Implement three similarity metrics from scratch in NumPy: cosine similarity, euclidean distance, and dot product (including batch forms used after candidate filtering).
5. Build exact brute-force *k*-nearest-neighbour search using batch matrix–vector math over the filtered embedding matrix rows.
6. Integrate `sentence-transformers` with the `all-MiniLM-L6-v2` model for 384-dimensional embeddings, with a `SimpleEmbeddingEngine` bag-of-words fallback when the neural stack is unavailable.
7. Maintain a three-way bridge among SQLite record identifiers, NumPy row indices, and the `id_mapping.json` ordered list so vectors and rows stay consistent.
8. Provide a CLI based on `argparse` and a Flask web UI with a session picker, shared file-processing paths for ingestion, and routes for search, insert, upload, history, stats, and JSON search where applicable.
9. Support file ingestion for TXT, CSV, and Excel with validation, header normalisation, format-aware chunking (including row-first tabular chunking with no overlap between rows except internal splits), and integration with batch insert.
10. Log user-level chat history for searches and inserts in the `messages` table — storing query text and summary fields such as metric, `top_k`, filters, counts, and timing — without storing large per-result blobs.
11. Detect and import legacy per-run storage from pre-v3.0 layouts (for example `minivecdb_data/` and `vectorstore_data/`) into the v3.0 `db_run/` layout via `VectorStore._maybe_migrate_legacy_storage`, preserving the original folders on disk so user data is never silently removed.
12. Supply automated tests (`pytest` and the standalone `tests/run_all_tests.py` runner) and a benchmarking approach grounded in NumPy-oriented performance measurement aligned with the batch search path.

---

## Chapter 4: Project Analysis and Design *(page 9)*

This chapter covers functional and environmental requirements, the entity–relationship design for MiniVecDB v3.0, how the database schema is defined and applied, normalisation to third normal form (with a note on the metadata EAV pattern), and representative sample data aligned with the implementation.

### 4.1: Hardware and Software Requirement Specifications (H/W and S/W requirements)

**Hardware**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86-64 or Apple Silicon, dual-core | Quad-core or better |
| RAM | 4 GB | 8 GB (embedding model loads approximately 80 MB; vectors also reside in RAM during use) |
| Disk | ~300 MB free | Additional space for growth (model cache ~80 MB, SQLite data, NumPy/runtime under `db_run/`) |
| Network | None required for normal operation | Optional; used on first run to download the embedding model if not already cached |
| GPU | Not required | — (sentence-transformers runs on CPU for this project) |

**Software**

| Layer | Requirement | Version |
|-------|-------------|---------|
| Operating system | macOS, Linux, or Windows | Current maintained releases |
| Language runtime | Python | 3.11 or newer |
| Dependencies | As listed in `requirements.txt` (see below) | Lower bounds per file |

```text
# MiniVecDB Dependencies
# Install with: pip install -r requirements.txt

# Core (required)
numpy>=1.24.0

# Embedding model (required for real embeddings)
sentence-transformers>=2.2.0

# Web interface (required for demo)
flask>=3.0.0

# File upload processing (CSV, Excel)
pandas>=2.0.0
openpyxl>=3.1.0

# Testing (development)
pytest>=7.0.0
```

The project also relies on Python built-in modules such as `sqlite3`, `json`, `os`, `time`, `uuid`, `argparse`, `dataclasses`, `typing`, `tempfile`, `re`, and `csv` for persistence, configuration, CLI, and glue logic without extra installs.

### 4.2: Use ER Diagrams

MiniVecDB v3.0 models the application with six entities: `sessions` (each run folder), `conversations` and `messages` (chat history of user queries), `collections` and `records` (scoped document storage), and `metadata` (EAV tags on records). Sessions anchor conversations and collections; conversations own messages; collections own records; records own metadata rows. Foreign keys use `ON DELETE CASCADE` so removing a session clears dependent rows through the graph.

**Figure 2 — MiniVecDB Entity-Relationship Diagram (v3.0)**

```text
┌──────────────────────────┐
│        sessions           │           ┌──────────────────────────────┐
│──────────────────────────│           │       conversations            │
│ PK id       INTEGER AUTO │──────1:N──│──────────────────────────────│
│    name     TEXT UNIQUE  │           │ PK id         INTEGER AUTO    │
│    storage_path TEXT     │           │ FK session_id  INTEGER        │
│    created_at   REAL     │           │    title       TEXT           │
│    last_used_at REAL     │           │    created_at  REAL           │
└──────────┬───────────────┘           └──────────────┬───────────────┘
           │                                          │
           │ 1:N                                      │ 1:N
           ▼                                          ▼
┌──────────────────────────┐           ┌──────────────────────────────┐
│      collections          │           │        messages                │
│──────────────────────────│           │──────────────────────────────│
│ PK id       INTEGER AUTO │           │ PK id              INT AUTO  │
│ FK session_id INTEGER    │           │ FK conversation_id INTEGER   │
│    name      TEXT        │           │    kind            TEXT      │
│    dimension  INTEGER    │           │    query_text      TEXT      │
│    description TEXT      │           │    metric          TEXT      │
│    created_at REAL       │           │    top_k           INTEGER   │
│  UNIQUE(session_id,name) │           │    category_filter TEXT      │
└──────────┬───────────────┘           │    result_count    INTEGER   │
           │                            │    elapsed_ms      REAL      │
           │ 1:N                        │    response_ref    TEXT      │
           ▼                            │    created_at      REAL      │
┌──────────────────────────┐           └──────────────────────────────┘
│       records             │
│──────────────────────────│
│ PK id       TEXT         │
│ FK session_id   INTEGER  │
│ FK collection_id INTEGER │
│    text         TEXT     │
│    created_at   REAL     │
└──────────┬───────────────┘
           │
           │ 1:N
           ▼
┌──────────────────────────┐
│       metadata            │
│──────────────────────────│
│ PK id       INTEGER AUTO │
│ FK record_id TEXT        │
│    key       TEXT        │
│    value     TEXT        │
└──────────────────────────┘
```

**Cardinalities**

| Parent | Child | Cardinality | FK Field |
|--------|-------|-------------|----------|
| sessions | conversations | 1..N | `conversations.session_id` |
| sessions | collections | 1..N | `collections.session_id` |
| sessions | records | 1..N | `records.session_id` |
| collections | records | 1..N | `records.collection_id` |
| conversations | messages | 1..N | `messages.conversation_id` |
| records | metadata | 1..N | `metadata.record_id` |

Deleting a `sessions` row cascades to related rows in `conversations` (and thus `messages`), `collections`, `records`, and `metadata`, so one session delete removes structured data across those five dependent areas as enforced by foreign keys. Separately, legacy pre-v3.0 storage folders such as `minivecdb_data/` and `vectorstore_data/` are imported into `db_run/` by `VectorStore._maybe_migrate_legacy_storage` on startup and are left intact on disk, so user data that predates the shared-DB layout is preserved outside the cascade rules of the current schema.

### 4.3: Table Creation Scripts and Database Schema

The schema lives in `ARCHITECTURE.py` as the `SCHEMA_SQL` constant. It is executed once at database connection time against the shared database at `db_run/minivecdb.db` (`CREATE TABLE IF NOT EXISTS ...`), with `PRAGMA foreign_keys = ON` so cascade deletes are enforced.

```sql
-- ----------------------------------------------------------------
-- 1) SESSIONS — one row per run folder under db_run/
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    storage_path  TEXT    NOT NULL,
    created_at    REAL    NOT NULL,
    last_used_at  REAL    NOT NULL
);
```

```sql
-- ----------------------------------------------------------------
-- 2) CONVERSATIONS — one default conversation per session
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    title       TEXT    NOT NULL DEFAULT 'Default conversation',
    created_at  REAL    NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

```sql
-- ----------------------------------------------------------------
-- 3) MESSAGES — user queries (search/insert) with response metadata
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL,
    kind             TEXT    NOT NULL CHECK (kind IN ('search','insert')),
    query_text       TEXT    NOT NULL,
    metric           TEXT    CHECK (metric IS NULL OR metric IN ('cosine','euclidean','dot')),
    top_k            INTEGER,
    category_filter  TEXT,
    result_count     INTEGER,
    elapsed_ms       REAL,
    response_ref     TEXT,
    created_at       REAL    NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
```

```sql
-- ----------------------------------------------------------------
-- 4) COLLECTIONS — session-scoped via composite UNIQUE
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS collections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    dimension   INTEGER NOT NULL DEFAULT 384,
    description TEXT    DEFAULT '',
    created_at  REAL    NOT NULL,
    UNIQUE (session_id, name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

```sql
-- ----------------------------------------------------------------
-- 5) RECORDS — one row per stored document; FK to session+collection
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS records (
    id             TEXT    PRIMARY KEY,
    session_id     INTEGER NOT NULL,
    collection_id  INTEGER NOT NULL,
    text           TEXT    NOT NULL,
    created_at     REAL    NOT NULL,
    FOREIGN KEY (session_id)    REFERENCES sessions(id)    ON DELETE CASCADE,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);
```

```sql
-- ----------------------------------------------------------------
-- 6) METADATA — EAV key/value tags on records
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metadata (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id  TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    value      TEXT    NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
);
```

```sql
-- ----------------------------------------------------------------
-- TRIGGERS
-- ----------------------------------------------------------------
-- Every new session gets a Default conversation row.
CREATE TRIGGER IF NOT EXISTS trg_create_default_conversation
AFTER INSERT ON sessions
BEGIN
    INSERT INTO conversations (session_id, title, created_at)
    VALUES (NEW.id, 'Default conversation', NEW.created_at);
END;

-- Every new session gets a "default" collection row.
CREATE TRIGGER IF NOT EXISTS trg_create_default_collection
AFTER INSERT ON sessions
BEGIN
    INSERT INTO collections (session_id, name, dimension, description, created_at)
    VALUES (NEW.id, 'default', 384, 'Default collection', NEW.created_at);
END;

-- Every new message bumps its session's last_used_at. Uses a
-- subquery to resolve session_id from the message's conversation.
CREATE TRIGGER IF NOT EXISTS trg_touch_session_on_message
AFTER INSERT ON messages
BEGIN
    UPDATE sessions
       SET last_used_at = NEW.created_at
     WHERE id = (SELECT session_id FROM conversations WHERE id = NEW.conversation_id);
END;
```

```sql
-- ----------------------------------------------------------------
-- INDEXES (9)
-- ----------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_sessions_last_used  ON sessions(last_used_at DESC);
CREATE INDEX IF NOT EXISTS idx_conv_session        ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_msg_conv            ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_msg_created         ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_collections_session ON collections(session_id);
CREATE INDEX IF NOT EXISTS idx_records_session     ON records(session_id);
CREATE INDEX IF NOT EXISTS idx_records_collection  ON records(collection_id);
CREATE INDEX IF NOT EXISTS idx_metadata_kv         ON metadata(key, value);
CREATE INDEX IF NOT EXISTS idx_metadata_record     ON metadata(record_id);
```

**Table 3 — Six-Table Schema Summary**

| # | Table | Primary Key | Key Foreign Keys | Purpose |
|---|-------|-------------|------------------|---------|
| 1 | `sessions` | `id` | — | One row per session/run folder; tracks storage path and recency |
| 2 | `conversations` | `id` | `session_id` → `sessions(id)` | Groups chat history; default row per session via trigger |
| 3 | `messages` | `id` | `conversation_id` → `conversations(id)` | Logs user search/insert actions and timing metadata |
| 4 | `collections` | `id` | `session_id` → `sessions(id)` | Namespaced collections per session (`UNIQUE(session_id, name)`) |
| 5 | `records` | `id` | `session_id` → `sessions(id)`, `collection_id` → `collections(id)` | Stores document text for embedded rows (vectors live outside SQL) |
| 6 | `metadata` | `id` | `record_id` → `records(id)` | EAV tags (`key`, `value`) for optional filtering |

### 4.4: Normalization (up to 3NF atleast)

The relational part of MiniVecDB is intended to stay in at least third normal form while supporting session isolation and flexible tagging.

**1NF.** Each column holds a single atomic value: there are no repeating groups or multi-valued columns in the six tables. The `metadata.value` column stores one scalar text value per row (one attribute value per EAV tuple), consistent with first normal form.

**2NF.** Every non-key attribute depends on the whole primary key, not on part of it. Tables use single-column primary keys (`INTEGER` surrogates where appropriate, `TEXT` for `records.id`). The `collections` table combines a surrogate `id` with `UNIQUE(session_id, name)` so collection naming is session-scoped without introducing partial dependencies on a composite natural key.

**3NF.** Transitive dependencies are avoided: for example, `records` does not duplicate `session_name` or `collection_name` (those would depend on `sessions` or `collections` via the foreign keys). Application code resolves names with `JOIN`s on `session_id` and `collection_id` when needed.

**EAV note.** The `metadata` table follows an Entity–Attribute–Value pattern (`record_id`, `key`, `value`). That trades strict uniform typing for flexibility: users can attach arbitrary keys without schema migrations. Values are stored as text ("stringly typed"), which is a deliberate product choice. Each EAV row still has a clear key `(id)` and functional dependencies: `(record_id, key, value)` avoids partial and transitive dependencies on non-key attributes at the row level.

Overall, MiniVecDB is designed in 3NF with a pragmatic EAV extension for metadata.

### 4.5: Sample Data

Illustrative benchmark and demo text comes from curated JSON shards under `data/generated/` (for example `technology.json`, `science.json`, `sports.json`, `health.json`, and `business.json`), together holding on the order of 150+ short documents across those topic areas. The rows below are representative only and do not dump the full corpus.

**`sessions`**

| id | name | storage_path | created_at | last_used_at |
|----|------|--------------|------------|--------------|
| 1 | demo_1713052800_a1b2c3 | db_run/demo_1713052800_a1b2c3 | 1713052800.0 | 1713052920.5 |
| 2 | demo_1713139200_f9e8d7c | db_run/demo_1713139200_f9e8d7c | 1713139200.0 | 1713139300.0 |

**`conversations`**

| id | session_id | title | created_at |
|----|------------|-------|------------|
| 1 | 1 | Default conversation | 1713052800.0 |
| 2 | 2 | Default conversation | 1713139200.0 |

**`messages`**

| id | conversation_id | kind | query_text | metric | top_k | category_filter | result_count | elapsed_ms | response_ref | created_at |
|----|-----------------|------|------------|--------|-------|-----------------|--------------|------------|--------------|------------|
| 1 | 1 | search | neural networks for image classification | cosine | 5 | NULL | 5 | 18.4 | NULL | 1713052850.0 |
| 2 | 1 | insert | CSV upload: health_articles.csv | NULL | NULL | NULL | 42 | 1250.0 | NULL | 1713052900.0 |

**`collections`**

| id | session_id | name | dimension | description | created_at |
|----|------------|------|-----------|-------------|------------|
| 1 | 1 | default | 384 |  | 1713052800.0 |
| 2 | 1 | papers | 384 | Research papers | 1713052810.0 |

**`records`**

| id | session_id | collection_id | text | created_at |
|----|------------|---------------|------|------------|
| vec_a1b2c3d4 | 1 | 2 | Deep learning has revolutionized computer vision and pattern recognition. | 1713052820.0 |

**`metadata`**

| id | record_id | key | value |
|----|-----------|-----|-------|
| 1 | vec_a1b2c3d4 | category | science |
| 2 | vec_a1b2c3d4 | source | sample |
| 3 | vec_a1b2c3d4 | year | 2023 |

**Vector Storage (outside SQL)**

Embeddings are not stored in SQLite. Each session folder holds `vectors.npy` (a float32 matrix of shape `(N, 384)`) and `id_mapping.json` (row index to `records.id`). The application may derive an in-memory inverse map from record id to row index.

```python
# vectors.npy  (shape (N, 384), float32)
_vectors[42] = np.array([0.12, -0.03, 0.45, ...], dtype=np.float32)

# id_mapping.json
_id_list[42] = "vec_a1b2c3d4"

# _id_to_index (derived in memory)
_id_to_index["vec_a1b2c3d4"] = 42
```

**`VectorRecord` (from `ARCHITECTURE.py`)**

```python
@dataclass
class VectorRecord:
    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]
    created_at: float
    collection: str = "default"
```

---

## Chapter 5: Work and Methodology *(page 17)*

**5.1 Layered Architecture.** MiniVecDB is organised as a thin presentation layer over a single orchestration class that coordinates three complementary stores. The CLI (`cli/main.py`) and the Flask web app (`web/app.py`) only parse input, call `VectorStore`, and render output; they do not embed vector-search logic. `VectorStore` (`core/vector_store.py`) is the engine: it invokes the embedding layer for text-to-vector conversion, persists structured rows through `DatabaseManager`, maintains an in-memory `(N, 384)` `float32` matrix in NumPy, and implements pre-filtering, batch scoring, and ranking. SQLite holds sessions, conversations, messages, collections, records, and EAV metadata; NumPy holds embeddings for fast batch math; the embedding module maps text to 384-dimensional vectors. The bridge file ties matrix rows to primary keys. This separation keeps relational integrity and query expressiveness in SQL while delegating similarity computation to NumPy.

**Figure 1 — Layered Architecture**

```text
┌─────────────────────────────────────────────────┐
│                   USER INTERFACE                 │
│           CLI (argparse) / Web (Flask)           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│              VectorStore (core engine)           │
│  insert() / search() / get() / delete() / etc.  │
└──┬──────────────┬──────────────────┬─────────────┘
   │              │                  │
   ▼              ▼                  ▼
┌────────┐  ┌──────────┐  ┌──────────────────────┐
│ SQLite │  │  NumPy   │  │  EmbeddingEngine     │
│ (DB)   │  │ (.npy)   │  │  (text → vectors)    │
└────────┘  └──────────┘  └──────────────────────┘
   │              │
   │    ┌─────────┘
   ▼    ▼
┌──────────────────┐
│  id_mapping.json │  ← Bridge file
│  (row ↔ ID)      │
└──────────────────┘
```

**5.2 Core Data Flows (INSERT / SEARCH / GET / DELETE).** All mutating and read paths are explicit so the hybrid design stays auditable. Insert embeds text, writes the record and metadata in SQLite, appends the new row to the matrix and ID list, then persists `vectors.npy` and `id_mapping.json`. Search embeds the query, optionally narrows candidate row indices via SQL metadata filtering, scores candidates with batch metric functions, sorts, and hydrates `SearchResult`s from SQLite. Get reads the relational row and EAV tags from SQLite and splices the vector row from NumPy via the index map. Delete removes the SQLite row (cascading metadata), deletes the matrix row, rebuilds the index map, and saves.

**Figure 3 — INSERT Data Flow**

```text
1. Input validation
   ├── Check text is non-empty string
   ├── Validate collection exists
   └── Check ID not duplicate

2. Text → Vector conversion
   └── EmbeddingEngine.encode("Python is great")
       → np.array([0.12, -0.03, ...], dtype=float32)  # shape (384,)

3. SQLite storage (atomic transaction)
   ├── INSERT INTO records (id, text, collection, created_at)
   └── INSERT INTO metadata (record_id, "topic", "programming")

4. NumPy storage (in-memory)
   ├── np.vstack([_vectors, new_vector])  # (N,384) → (N+1,384)
   └── _id_list.append("vec_a1b2c3d4")

5. Disk persistence
   ├── np.save("vectors.npy", _vectors)
   └── json.dump(_id_list, "id_mapping.json")

6. Return record ID → "vec_a1b2c3d4"
```

**Figure 4 — SEARCH Data Flow**

```text
1. Input validation
   ├── Check database not empty
   └── Validate metric name

2. Embed the query
   └── EmbeddingEngine.encode("artificial intelligence")
       → query_vector: shape (384,)

3. Pre-filter candidates (SQL)
   ├── SELECT record_id FROM metadata WHERE key='year' AND CAST(value AS REAL) > 2020
   ├── Intersect with collection IDs (if collection specified)
   └── Convert matching IDs → NumPy row indices

4. Batch similarity computation (NumPy)
   ├── candidate_vectors = _vectors[filtered_indices]  # shape (M, 384)
   ├── scores = batch_cosine_similarity(query_vector, candidate_vectors)  # shape (M,)
   └── This uses: candidate_vectors @ query_vector (matrix × vector)

5. Ranking
   ├── sorted_positions = np.argsort(scores)[::-1]  # descending
   └── top_k_positions = sorted_positions[:3]

6. Build results
   ├── For each top result:
   │   ├── Look up record ID from _id_list
   │   ├── Fetch record from SQLite (text, metadata)
   │   └── Build SearchResult(record, score, rank, metric)
   └── Return [SearchResult, SearchResult, SearchResult]
```

GET: fetch id/text/collection/timestamp and metadata from SQLite; resolve the row index via `_id_to_index`; take `_vectors[idx]`; assemble a `VectorRecord`. DELETE: confirm existence, locate the row index, delete from SQLite (metadata cascades), remove the NumPy row and list entry, rebuild `_id_to_index`, persist.

**5.3 The Three-Way Bridge.** Each stored vector sits at row `i` of the `(N, 384)` matrix `_vectors`. The `records` table holds authoritative text and keys records by string id. `id_mapping.json` stores the ordered list `_id_list` such that `_id_list[i]` is the SQLite primary key for row `i`. A reverse dictionary maps id → index for O(1) lookups. Any operation that changes row order or membership must update both the matrix row, the ordered list, and the reverse map together before saving.

**Figure 5 — Three-Way Bridge (SQLite / NumPy / JSON)**

```text
SQLite:     records table has "vec_a1b2c3d4" at row id='vec_a1b2c3d4'
NumPy:      _vectors has the 384-dim array at row index 42
JSON:       _id_list[42] = "vec_a1b2c3d4"
                └── This is the BRIDGE

_id_to_index = {"vec_a1b2c3d4": 42}   ← Reverse lookup (O(1) by ID)
```

**5.4 Similarity Search Engine (Built from Scratch).** All metrics are implemented in `core/distance_metrics.py` with batch variants used at query time. Cosine similarity measures alignment of direction: A·B / (‖A‖·‖B‖), range [-1, 1], best 1.0. Euclidean distance is √(Σ(Aᵢ − Bᵢ)²), range [0, ∞), best 0.0. Dot product is Σ(Aᵢ·Bᵢ), range (-∞, ∞); higher is better, and it matches cosine when vectors are normalised. The cosine and dot-product batch paths reduce to a single matrix–vector multiply, `candidate_vectors @ query_vector`, which NumPy dispatches to BLAS. The euclidean batch path instead broadcasts `candidate_vectors - query_vector` and takes a row-wise norm. All three metrics are O(N · D) per query for N candidates and dimension D = 384.

**Table 2 — Distance Metrics Comparison**

| Metric | Formula | Range | Best Score | Use Case | Implementation |
|--------|---------|-------|------------|----------|----------------|
| Cosine similarity | A·B / (‖A‖·‖B‖) | [-1, 1] | 1.0 | Text and embeddings where length varies | `batch_cosine_similarity` — dot products, row norms, safe divide, clip |
| Euclidean distance | √(Σ(Aᵢ − Bᵢ)²) | [0, ∞) | 0.0 | When magnitude matters in the embedding space | `batch_euclidean_distance` — broadcast subtract, row norms |
| Dot product | Σ(Aᵢ·Bᵢ) | (-∞, ∞) | Higher is better (often with normalised vectors) | Fast path when norms are implicit | `batch_dot_product` — matrix–vector multiply only |

**5.5 Embedding Engine.** Production quality uses `sentence-transformers` with `all-MiniLM-L6-v2`, producing 384-dimensional `float32` vectors; weights cache under `db_run/model_cache/huggingface`. Lazy loading avoids paying model startup until `encode()` runs. When the neural stack is unavailable, the factory `create_embedding_engine(fallback=True)` instantiates `SimpleEmbeddingEngine(dimension=384)` — a deterministic hash-based bag-of-words engine configured to match the neural engine's dimension — so CI and offline runs keep producing 384-d `float32` vectors and remain meaningful.

**5.6 Pre-Filtering with SQL.** Search is two-stage: optional SQL filters on `session_id`, collection, and EAV `(key, value)` pairs yield a candidate id set; those ids convert to row indices via `_id_to_index`. Only that subset participates in `candidate_vectors @ query_vector` and sorting. Example: restrict `category` to `science`, then rank the remaining rows by cosine similarity — metadata work stays in the relational engine, vector work stays in NumPy.

**5.7 File Upload Pipeline (TXT / CSV / Excel).** `core/file_processor.py` validates inputs: maximum size 10 MB and extensions `.txt`, `.csv`, `.xlsx`, `.xls`. CSV/Excel paths detect likely header rows, skip blank noise, deduplicate repeated header-like rows, and serialise each logical row as a stable `Column: value | …` string. TXT uses sentence/word chunking with configurable overlap; tabular formats chunk row-first with zero overlap, splitting an oversized row only within that row. Output is `(texts, metadata_list)` consumed directly by `insert_batch()`.

**5.8 Session Management and Shared DB (v3.0).** Before v3.0, each session folder could carry its own `minivecdb.db`. v3.0 stores all sessions in one shared `db_run/minivecdb.db`, keying `records` and `collections` with `session_id` foreign keys. Per-session folders hold only `vectors.npy` and `id_mapping.json`; `.active_run` names the currently bound session directory under `db_run/`.

**5.9 Legacy Storage Migration.** On startup, `VectorStore._maybe_migrate_legacy_storage` detects older per-run artefacts (`minivecdb_data/`, `vectorstore_data/`) under the repository root and copies their vectors, id mappings, and SQLite data into `db_run/` so previously recorded text and embeddings remain addressable inside the v3.0 shared-DB layout. The copy path never unlinks user data from the legacy folders: originals stay on disk untouched, which is the intended safety stance for a learning project where reproducibility of earlier experiments matters.

**5.10 Testing and Benchmarking.** Tests live under `tests/` (distance metrics, embeddings, vector store, database, session schema with triggers and cascades, edge cases, integration). `tests/run_all_tests.py` offers a pytest-free runner. Performance experiments use `tests/benchmark.py` against the curated corpus (insert throughput, per-query latency distributions, memory footprint, and per-metric cost at full corpus size).

---

## Chapter 6: Results and Discussion *(page 21)*

**6.1 Dataset.** The evaluation corpus is approximately 151 curated documents spread across five JSON shards in `data/generated/`: Technology, Science, Sports, Health, and Business — covering AI, programming, physics, biology, sports, medicine, finance, and related subtopics. The loader `data/sample_dataset.py` concatenates `technology.json`, `science.json`, `sports.json`, `health.json`, and `business.json` into a single list of `{text, metadata}` objects for demos and benchmarks.

**6.2 Benchmark Results.** The benchmark driver (`tests/benchmark.py`) uses corpus sizes `SIZES = [25, 50, 100, 151]` and `METRICS = ["cosine", "euclidean", "dot"]`. All three metrics are exercised in the dedicated `bench_metric_comparison` phase; the generic `bench_query_latency_*` phases fix `metric='cosine'` across a 50-query workload. The report-level qualitative expectations are that a SimpleEmbeddingEngine-only run completes on the order of seconds because its hashing path is cheap, while runs that include SentenceTransformer encoding are dominated by neural batch encoding and take substantially longer per size; these are illustrative, not hard-coded timing bounds in `tests/benchmark.py`. Metric comparison typically shows dot product as the fastest (pure matrix–vector multiply), cosine as intermediate (dot plus norms and divide), and euclidean as slowest (broadcast subtract, squared sum, square root) — a qualitative ordering rather than a fixed latency table. Memory scales linearly with stored rows: the float32 matrix occupies `151 × 384 × 4 = 231,936` bytes for the full corpus, and the benchmark actually records `store._vectors.nbytes` via `numpy.ndarray.nbytes` rather than a hard-coded value. Treat absolute insertion and search numbers as environment-dependent unless captured from your own `tests/benchmark_results.json` run.

**Table 5 — Benchmark Summary (qualitative; measure your own run for exact numbers)**

| Operation | Engine | Dataset Size | Metric / Mode | Result |
|-----------|--------|--------------|---------------|--------|
| End-to-end benchmark (insert + search phases) | SimpleEmbeddingEngine | 25 – 151 corpus | all phases | qualitative — hashing is cheap; wall-clock dominated by I/O and sorting |
| End-to-end benchmark (insert + search phases) | SentenceTransformer | 25 – 151 corpus | all phases | qualitative — wall-clock dominated by neural encoding; substantially slower than Simple |
| Vector matrix memory | Either engine | 151 | N/A | `151 × 384 × 4 = 231,936` bytes, read at runtime via `numpy.ndarray.nbytes` |
| Metric-comparison phase | Either engine | 151 | cosine vs euclidean vs dot | qualitative — dot fastest, cosine intermediate, euclidean slowest |
| Query-latency phase | Either engine | 25 – 151 | `metric='cosine'` only (50-query workload × `NUM_RUNS`) | qualitative — sub-10 ms NumPy phase at N ≈ 150 on typical hardware |
| Batch insert throughput | Either | 25 – 151 | grouped `insert_batch` | qualitative — encoder-bound; docs/sec is machine- and cache-dependent |

Brute-force kNN is O(N·D) per query after candidate selection. For N ≈ 150, documented example latency summaries sit in the sub–10 ms class on typical hardware for the NumPy phase alone; the linear scan eventually dominates as N grows, motivating approximate nearest-neighbour structures in future work.

**6.3 Semantic Search Demo.** `demo/semantic_search.py` walks through loading the corpus, creating a fresh `VectorStore` with `new_run=True`, bulk inserting all documents via `insert_batch()`, running diverse natural-language queries with printed top-k hits, applying a metadata filter (for example searching within the Science stream only), printing aggregate statistics, and finally comparing paraphrased query pairs to show overlap in retrieved ids. An illustrative natural query such as "machine learning for medical imaging" can surface both Health and Technology articles because shared-transformer embeddings align on meaning rather than term overlap — top results cohere with the query's intent across categories.

**6.4 SQL Techniques Verified.**

**Table 4 — SQL Techniques Demonstrated**

| # | Technique | Demonstrated In (SQL query name or trigger) | File |
|---|-----------|----------------------------------------------|------|
| 1 | LEFT JOIN | `list_collections_in_session`, `stats_per_collection` (`collections` LEFT JOIN `records`) | `ARCHITECTURE.py` (`SQL_QUERIES`) |
| 2 | INNER JOIN | `history_for_session` (`messages` JOIN `conversations`), `filter_by_metadata` (`metadata` JOIN `records`) | `ARCHITECTURE.py` (`SQL_QUERIES`) |
| 3 | Scalar / correlated subquery | `trg_touch_session_on_message` (`SELECT session_id FROM conversations WHERE id = NEW.conversation_id`) | `ARCHITECTURE.py` (`SCHEMA_SQL`) |
| 4 | Aggregate + GROUP BY | `stats_per_collection`, `list_collections_in_session` (`COUNT(*) ... GROUP BY collection_id`) | `ARCHITECTURE.py` (`SQL_QUERIES`) |
| 5 | Plain aggregate (COUNT) | `count_records_in_session`, `count_messages_in_session` (driven by `DatabaseManager.list_sessions()`) | `ARCHITECTURE.py` + `storage/database.py` |
| 6 | Triggers (3) | `trg_create_default_conversation`, `trg_create_default_collection`, `trg_touch_session_on_message` | `ARCHITECTURE.py` (`SCHEMA_SQL`) |
| 7 | CHECK constraint | `messages.kind IN ('search','insert')` and `messages.metric IN ('cosine','euclidean','dot')` | `ARCHITECTURE.py` |
| 8 | Composite UNIQUE | `collections(session_id, name)` | `ARCHITECTURE.py` |
| 9 | UPSERT (ON CONFLICT) | `upsert_session` template, executed via `DatabaseManager._ensure_session()` | `ARCHITECTURE.py` + `storage/database.py` |
| 10 | Cascade DELETE | all foreign-key relationships (`ON DELETE CASCADE` in every child table) | `ARCHITECTURE.py` (`SCHEMA_SQL`) |

**6.5 Discussion.** MiniVecDB trades the scale-out features of production vector databases — no ANN index, no distributed shards — for transparency: every student-visible operation maps to SQL rows or explicit NumPy math. The hybrid pattern plays to each layer's strengths: declarative filters and joins in SQLite are inexpensive for modest N, while dense matrix–vector products exploit cache-friendly BLAS kernels. Limits include O(N) exhaustive search, single-writer concurrency assumptions suitable for coursework, and dependence on re-embedding when rebuilding vectors from text after corruption. These boundaries clarify what "from scratch" buys pedagogically versus what would change in an industrial deployment.

---

## Chapter 7: Conclusion *(page 24)*

MiniVecDB delivers a complete miniature data platform that unifies classical relational practice with modern retrieval. The SQLite schema normalises sessions, conversations, message history, collections, records, and metadata with third-normal-form structure, `ON DELETE CASCADE` integrity, and triggers that maintain default child rows and session activity timestamps — showing how declarative rules keep multi-table state consistent without application-level glue.

The retrieval layer implements exact k-nearest-neighbour scoring in `float32` with three transparent metrics and batch kernels built on NumPy, avoiding black-box similarity libraries for the core math. Students can read a query path end-to-end: embed text, map SQL-filtered ids to matrix rows, multiply, sort, hydrate rows.

The hybrid storage pattern binds these pieces: SQLite remains the source of truth for text and tags; NumPy stores the fast path for embeddings; `id_mapping.json` is the bridge that keeps row order aligned with string keys after inserts and deletes. Runtime recovery can rebuild vectors from text if artefacts diverge, reinforcing SQLite's authority.

Three learning outcomes stand out: (1) relational modelling with automation through triggers and cascades; (2) implementing and benchmarking exact KNN with explicit complexity; (3) composing SQL plus ndarray storage plus JSON into one coherent API. In one line: MiniVecDB shows that the gap between classical DBMS craft and modern vector search is narrower than it appears, once you construct both halves from first principles.

---

## Chapter 8: Future Scope of Work *(page 25)*

- **Approximate Nearest Neighbour indexing (IVF, HNSW).** Replace the linear scan with partition trees or graph indexes so sublinear query time scales to millions of vectors while trading a controlled amount of recall.
- **Multi-conversation threading per session.** The schema already models multiple conversations; exposing thread pickers and routing in the web UI would complete the story for long-lived research notebooks.
- **Authentication and role-based access control.** Shared `minivecdb.db` deployments need user identities, scoped sessions, and auditability for classroom or lab servers.
- **Hybrid ranking: BM25 plus vectors.** Combine SQLite FTS5 lexical scoring with cosine similarity for queries where keywords and semantics must agree — reducing false positives from either signal alone.
- **Vector quantisation (PQ / scalar quantisation).** Shrink on-disk `(N, 384)` float matrices for large corpora while controlling reconstruction error during search.
- **Richer ingest: PDF and DOCX.** Extend `file_processor`-style validation and text extraction pipelines beyond tabular and plain text for real office workloads.
- **Portable session archives.** Export and import a session's SQLite slice plus `vectors.npy`, `id_mapping.json`, and manifest for backup and hand-off between machines.
- **GPU-accelerated embedding batches.** Optional PyTorch-backed batching for `encode_batch` when hardware is available, shaving wall-clock without changing the API.
- **Observability.** Add query timing breakdowns, slow-query logs, and CI hooks that regression-test latency and accuracy against fixed corpora so changes to metrics or schema stay measurable.

---

## REFERENCES

1. Reimers, N., and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *Proceedings of EMNLP*, 2019.
2. Hugging Face Model Card — sentence-transformers/all-MiniLM-L6-v2. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
3. NumPy Reference Documentation. https://numpy.org/doc/stable/
4. SQLite Documentation — CREATE TABLE / Triggers / Foreign Keys. https://sqlite.org/docs.html
5. Flask Documentation. https://flask.palletsprojects.com/
6. pandas Documentation. https://pandas.pydata.org/docs/
7. openpyxl Documentation. https://openpyxl.readthedocs.io/
8. Johnson, J., Douze, M., Jégou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 2019 (FAISS).
9. Pinecone Vector Database Documentation. https://docs.pinecone.io/
10. ChromaDB Documentation. https://docs.trychroma.com/
11. Weaviate Documentation. https://weaviate.io/developers/weaviate
12. Codd, E. F. "A Relational Model of Data for Large Shared Data Banks." *Communications of the ACM*, 1970.
13. pytest Documentation. https://docs.pytest.org/
14. MiniVecDB project source — `AGENTS.md` and `README.md` (internal project files).
