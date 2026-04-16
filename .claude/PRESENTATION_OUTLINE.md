# MiniVecDB — Presentation Outline (PPT Content Structure)

> Slide-by-slide content outline for the MiniVecDB course presentation.
> Every factual claim is grounded in the MiniVecDB codebase (`ARCHITECTURE.py`, `core/*.py`, `storage/*.py`, `cli/main.py`, `web/app.py`, `tests/benchmark_results.json`) and in the companion academic write-up at `.claude/PROJECT_REPORT.md`.
>
> Structure: 8 sections (matching the required Table of Contents exactly, in order), 28 slides total. Each slide gives a title, 4–6 bullet points, and — where useful — an ASCII/diagram block, a code reference, or a screenshot placeholder with capture instructions.
>
> TOC (locked):
> 1. Introduction & Motivation
> 2. Problem Statement
> 3. ER-Diagram
> 4. Technology Stack
> 5. Database Schema and Normalization
> 6. Project Implementation (code/screenshots etc.)
> 7. Project Results
> 8. References

---

## 1. Introduction & Motivation

### Slide 1 — Title slide

- **Project**: MiniVecDB — A Mini Vector Database Built From Scratch (v3.0)
- **Course**: Database Management Systems (DBMS) — capstone project
- **Language / Core tech**: Python 3.11+, SQLite, NumPy, sentence-transformers, Flask
- **Hard constraint**: no ChromaDB, FAISS, Pinecone, or Weaviate — vector engine implemented from first principles
- **Team / author / roll no.**: _[placeholder]_
- **Date**: _[placeholder]_

### Slide 2 — What is MiniVecDB?

- A hybrid-storage mini vector database written entirely in Python
- **Dual-purpose** design:
  - Classical DBMS: 3NF schema, SQL, JOINs, triggers, CHECK/UNIQUE constraints, CASCADE deletes
  - Modern AI retrieval: 384-dim text embeddings + exact brute-force *k*-NN over them
- **Source of truth**: a single shared SQLite database at `db_run/minivecdb.db`
- **Vector artefacts** live per-session on disk: `vectors.npy` + `id_mapping.json`
- Ships with a CLI (argparse, 10 subcommands), a Flask web UI (session picker + 12 routes), and bulk file ingestion for TXT / CSV / Excel

### Slide 3 — Motivation: exact match vs semantic match

- Relational databases excel at **exact** retrieval: `WHERE name = 'Alice'`
- But users often want to retrieve by **meaning**: *"machine learning algorithms"* should return documents about neural networks, deep learning, and AI even when those exact words never appear
- This shift — symbolic equality → statistical similarity — mirrors ML's shift from rule-based to deep models
- Illustration:
  - "The cat sat on the mat" → `[0.12, -0.03, 0.45, …]`
  - "A kitten rested on a rug" → very similar vector
  - "Stock market crashed today" → very different vector
- MiniVecDB exposes this mechanism end to end instead of hiding it inside a black-box library

### Slide 4 — Core idea: hybrid storage (three stores, one system)

- **SQLite** → all structured data (sessions, conversations, messages, collections, records, EAV metadata); `PRAGMA foreign_keys = ON`
- **NumPy (.npy)** → a single `(N, 384) float32` matrix per session; enables batch similarity via `matrix @ query_vector`
- **JSON bridge** → `id_mapping.json` stores an ordered list so `_id_list[i]` is the SQLite record ID of the vector at row `i`
- SQL does the **filtering** (metadata, collection, session isolation); NumPy does the **math** (batch scoring, ranking)
- No vector DB library used for the retrieval core — only NumPy BLAS under `matrix @ vector`

---

## 2. Problem Statement

### Slide 5 — Problem framing

- A DBMS capstone should exercise **both** classical relational skills and modern retrieval patterns that students see in AI-backed apps
- Course-friendly vector stacks (FAISS / Chroma / Pinecone / Weaviate) typically **hide** indexing, distance computation, and ranking
- That accelerates prototyping but **obscures** the database-adjacent mechanics the DBMS course is meant to teach
- The project must: (a) unify a rigorous relational substrate with an explicit vector pipeline; (b) preserve session isolation in SQL; (c) never silently delete user data; (d) be readable line-by-line
- Goal: produce a compact, end-to-end artefact where every operation maps to SQL rows or explicit NumPy math

### Slide 6 — Objectives (12, condensed)

1. 3NF SQLite schema with **6 tables**: `sessions`, `conversations`, `messages`, `collections`, `records`, `metadata`
2. FK integrity with `ON DELETE CASCADE` across the entire relationship graph
3. Session-bound CRUD via a single `DatabaseManager` class
4. Demonstrate SQL techniques: **INNER/LEFT JOIN, subquery, aggregate + GROUP BY, CHECK, composite UNIQUE, UPSERT (ON CONFLICT), CASCADE, triggers**
5. **Three** similarity metrics from scratch in NumPy: **cosine, euclidean, dot** (single + batch forms)
6. **Exact brute-force *k*-NN** over filtered candidates — no ANN index
7. `sentence-transformers` with `all-MiniLM-L6-v2` (384-d) + `SimpleEmbeddingEngine` bag-of-words fallback
8. Three-way bridge among SQLite IDs, NumPy row indices, and `id_mapping.json`
9. `argparse` CLI + Flask web UI (session picker, search, insert, upload, history, stats, records)
10. File ingest pipeline for TXT / CSV / Excel with header normalisation + row-first chunking (`core/file_processor.py`)
11. Chat-style history logging for user queries only (no large result blobs)
12. Legacy storage migration + full automated test suite (`pytest` + `tests/run_all_tests.py`) and a benchmark driver (`tests/benchmark.py`)

---

## 3. ER-Diagram

### Slide 7 — Six entities and their cardinalities

- **sessions** — one row per run folder under `db_run/` (`UNIQUE name`)
- **conversations** — chat threads; default row per session auto-created by trigger
- **messages** — user queries (`kind IN ('search','insert')`) with response metadata
- **collections** — namespaced record groups per session (`UNIQUE(session_id, name)`)
- **records** — stored documents (FK to both `sessions` and `collections`)
- **metadata** — EAV key/value tags per record

| Parent | Child | Cardinality |
|--------|-------|-------------|
| sessions | conversations | 1..N |
| sessions | collections | 1..N |
| sessions | records | 1..N |
| collections | records | 1..N |
| conversations | messages | 1..N |
| records | metadata | 1..N |

### Slide 8 — ER diagram (visual)

```text
┌──────────────────────────┐
│        sessions           │           ┌──────────────────────────────┐
│──────────────────────────│           │        conversations           │
│ PK id         INT AUTO   │──────1:N──│──────────────────────────────│
│    name       TEXT UNIQ  │           │ PK id          INT AUTO       │
│    storage_path TEXT     │           │ FK session_id  INTEGER        │
│    created_at   REAL     │           │    title       TEXT           │
│    last_used_at REAL     │           │    created_at  REAL           │
└──────────┬───────────────┘           └──────────────┬───────────────┘
           │                                          │
           │ 1:N                                      │ 1:N
           ▼                                          ▼
┌──────────────────────────┐           ┌──────────────────────────────┐
│      collections          │           │          messages              │
│──────────────────────────│           │──────────────────────────────│
│ PK id         INT AUTO   │           │ PK id              INT AUTO  │
│ FK session_id INTEGER    │           │ FK conversation_id INTEGER   │
│    name       TEXT       │           │    kind            TEXT CHK  │
│    dimension  INTEGER    │           │    query_text      TEXT      │
│    description TEXT      │           │    metric          TEXT CHK  │
│    created_at REAL       │           │    top_k           INTEGER   │
│  UNIQUE(session_id,name) │           │    category_filter TEXT      │
└──────────┬───────────────┘           │    result_count    INTEGER   │
           │                           │    elapsed_ms      REAL      │
           │ 1:N                       │    response_ref    TEXT      │
           ▼                           │    created_at      REAL      │
┌──────────────────────────┐           └──────────────────────────────┘
│        records            │
│──────────────────────────│
│ PK id         TEXT       │
│ FK session_id    INTEGER │
│ FK collection_id INTEGER │
│    text          TEXT    │
│    created_at    REAL    │
└──────────┬───────────────┘
           │ 1:N
           ▼
┌──────────────────────────┐
│       metadata            │
│──────────────────────────│
│ PK id        INT AUTO    │
│ FK record_id TEXT        │
│    key       TEXT        │
│    value     TEXT        │
└──────────────────────────┘
```

- Source: `SCHEMA_SQL` in `ARCHITECTURE.py`
- All child tables declare `ON DELETE CASCADE`

### Slide 9 — Cascade delete chain

```text
DELETE session "demo_123"
  ├→ DELETE conversations WHERE session_id = X          (cascade from sessions)
  │    └→ DELETE messages WHERE conversation_id IN (…) (cascade from conversations)
  ├→ DELETE collections WHERE session_id = X            (cascade from sessions)
  │    └→ DELETE records WHERE collection_id IN (…)     (cascade from collections)
  │         └→ DELETE metadata WHERE record_id IN (…)   (cascade from records)
  └→ (Also: DELETE records WHERE session_id = X — dual FK cascade)
```

- One `DELETE FROM sessions WHERE id = ?` removes structured data across **five** dependent tables
- Enforced at the SQLite engine level via `PRAGMA foreign_keys = ON` (set in `DatabaseManager.__init__`)
- Legacy storage (`minivecdb_data/`, `vectorstore_data/`) is **imported** into `db_run/` by `VectorStore._maybe_migrate_legacy_storage`, but the originals are **never unlinked** — user data is preserved

---

## 4. Technology Stack

### Slide 10 — Stack overview

| Layer | Technology | Role in MiniVecDB |
|-------|------------|-------------------|
| Language | Python 3.11+ | Core implementation, type hints everywhere |
| Structured storage | `sqlite3` (built-in) | Shared relational DB at `db_run/minivecdb.db` |
| Vector storage | NumPy | `(N, 384) float32` matrices, batch matrix–vector math |
| Embedding model | `sentence-transformers` `all-MiniLM-L6-v2` | Text → 384-d vector (with `SimpleEmbeddingEngine` bag-of-words fallback) |
| File processing | `csv` + `pandas` + `openpyxl` | TXT / CSV / Excel ingest and normalisation |
| CLI | `argparse` (built-in) | 10 subcommands, session-aware |
| Web UI | Flask + Jinja2 | Session picker + HTML routes + `/api/search` JSON endpoint |
| Testing | `pytest` + `tests/run_all_tests.py` | Full suite and a standalone runner |
| Bridge file | JSON | `id_mapping.json` ties NumPy row index ↔ SQLite record ID |

- Hybrid architecture principle: SQL for what SQL is best at; NumPy for batch math; JSON as the glue

### Slide 11 — Hardware / software requirements

- **Hardware**
  - CPU: x86-64 or Apple Silicon, dual-core minimum (quad-core recommended)
  - RAM: 4 GB minimum / 8 GB recommended (embedding model ≈ 80 MB + vectors in RAM)
  - Disk: ~300 MB free (plus growth for `db_run/`)
  - GPU: not required — sentence-transformers runs on CPU for this project
- **Software**
  - OS: macOS, Linux, or Windows (current maintained releases)
  - Python: 3.11 or newer
  - Network: only for first-time download of the 80 MB model; offline thereafter
- **`requirements.txt` (lower bounds)**

```text
numpy>=1.24.0
sentence-transformers>=2.2.0
flask>=3.0.0
pandas>=2.0.0
openpyxl>=3.1.0
pytest>=7.0.0
```

- Also uses Python built-ins: `sqlite3`, `json`, `os`, `time`, `uuid`, `argparse`, `dataclasses`, `typing`, `tempfile`, `re`, `csv`

---

## 5. Database Schema and Normalization

### Slide 12 — Six-table schema summary

| # | Table | Primary Key | Foreign Keys | Purpose |
|---|-------|-------------|---------------|---------|
| 1 | `sessions` | `id INTEGER` | — | One row per session / run folder; tracks storage path + recency |
| 2 | `conversations` | `id INTEGER` | `session_id` → `sessions(id)` | Groups chat history; default row per session via trigger |
| 3 | `messages` | `id INTEGER` | `conversation_id` → `conversations(id)` | Logs user search/insert actions + timing metadata |
| 4 | `collections` | `id INTEGER` | `session_id` → `sessions(id)` | Namespaced collections per session (`UNIQUE(session_id, name)`) |
| 5 | `records` | `id TEXT` (e.g. `vec_a1b2c3d4`) | `session_id`, `collection_id` | Document text rows; vectors live outside SQL |
| 6 | `metadata` | `id INTEGER` | `record_id` → `records(id)` | EAV tags `(key, value)` for optional filtering |

- Schema lives in `SCHEMA_SQL` (`ARCHITECTURE.py`), executed idempotently via `conn.executescript(...)` with `CREATE TABLE IF NOT EXISTS`
- 9 supporting indexes (e.g. `idx_sessions_last_used`, `idx_metadata_kv`, `idx_records_session`) documented in the same file

### Slide 13 — Three triggers + domain constraints

- **`trg_create_default_conversation`** — `AFTER INSERT ON sessions` → inserts a "Default conversation" row
- **`trg_create_default_collection`** — `AFTER INSERT ON sessions` → inserts a `"default"` collection with `dimension = 384`
- **`trg_touch_session_on_message`** — `AFTER INSERT ON messages` → uses a **subquery** to hop `conversation_id → session_id` and bump `sessions.last_used_at`

```sql
CREATE TRIGGER IF NOT EXISTS trg_touch_session_on_message
AFTER INSERT ON messages
BEGIN
    UPDATE sessions
       SET last_used_at = NEW.created_at
     WHERE id = (SELECT session_id FROM conversations WHERE id = NEW.conversation_id);
END;
```

- **CHECK constraints**:
  - `messages.kind IN ('search','insert')`
  - `messages.metric IN ('cosine','euclidean','dot')` (nullable)
- **Composite UNIQUE**: `collections(session_id, name)` — two sessions can each own a collection named `"papers"` without collision
- **UPSERT**: `INSERT ... ON CONFLICT(name) DO UPDATE SET storage_path = excluded.storage_path` for idempotent session registration (`SQL_QUERIES["upsert_session"]`)

### Slide 14 — SQL techniques demonstrated

| # | Technique | Where in the codebase |
|---|-----------|-----------------------|
| 1 | **LEFT JOIN** | `SQL_QUERIES["list_collections_in_session"]`, `stats_per_collection` (collections LEFT JOIN records) |
| 2 | **INNER JOIN** | `history_for_session` (messages JOIN conversations), `filter_by_metadata` (metadata JOIN records), `get_record`, `list_records` |
| 3 | **Scalar subquery** | Trigger `trg_touch_session_on_message` — `SELECT session_id FROM conversations WHERE id = NEW.conversation_id` |
| 4 | **Aggregate + GROUP BY** | `stats_per_collection`, `list_collections_in_session` (`COUNT(*) … GROUP BY collection_id`) |
| 5 | **Plain aggregate (COUNT)** | `count_records_in_session`, `count_messages_in_session` |
| 6 | **Triggers (3)** | See Slide 13 — all in `SCHEMA_SQL` |
| 7 | **CHECK constraint** | `messages.kind`, `messages.metric` |
| 8 | **Composite UNIQUE** | `collections(session_id, name)` |
| 9 | **UPSERT (ON CONFLICT)** | `SQL_QUERIES["upsert_session"]`, called by `DatabaseManager._ensure_session` |
| 10 | **Cascade DELETE** | Every FK relationship declares `ON DELETE CASCADE` |

- All 35 SQL templates live centrally in `SQL_QUERIES` (`ARCHITECTURE.py`) and are always executed with parameterised `?` placeholders — no string formatting of user input

### Slide 15 — Normalization: 1NF, 2NF, 3NF (+ EAV note)

- **1NF** — every column is atomic: no arrays, no JSON blobs, no multi-valued fields. `metadata.value` holds one scalar text value per row
- **2NF** — every non-key attribute depends on the **full** primary key. All six tables use single-column surrogate PKs (`INTEGER AUTOINCREMENT` or `TEXT` for `records.id`). The `collections` table adds a composite `UNIQUE(session_id, name)` so naming is session-scoped without introducing partial dependencies on a natural composite key
- **3NF** — no transitive dependencies. `records` never duplicates `session_name` or `collection_name` — those are resolved via JOINs on `session_id` / `collection_id`. Names live only in their owning tables
- **EAV note** — the `metadata` table follows the Entity-Attribute-Value pattern (`record_id`, `key`, `value`). This trades strict typing for flexibility: users can attach arbitrary keys without schema migrations. A pragmatic 3NF extension, not a violation of it
- **Net result**: the relational layer is **3NF with an intentional EAV extension** for user-defined tags

---

## 6. Project Implementation (code/screenshots etc.)

### Slide 16 — Layered architecture + three-way bridge

```text
┌────────────────────────────────────────────────────┐
│                   USER INTERFACE                    │
│           CLI (argparse) / Web (Flask)              │
└────────────────────┬───────────────────────────────┘
                     │
┌────────────────────▼───────────────────────────────┐
│              VectorStore (core engine)              │
│  insert() / search() / get() / delete() / update() │
└──┬──────────────┬──────────────────┬───────────────┘
   │              │                  │
   ▼              ▼                  ▼
┌────────┐  ┌──────────┐  ┌─────────────────────┐
│ SQLite │  │  NumPy   │  │  EmbeddingEngine    │
│ (DB)   │  │ (.npy)   │  │  (text → vectors)   │
└────────┘  └──────────┘  └─────────────────────┘
   │              │
   │    ┌─────────┘
   ▼    ▼
┌──────────────────┐
│ id_mapping.json  │  ← bridge file (row ↔ ID)
└──────────────────┘
```

- **Three-way bridge**:
  - `_vectors[42]` → the 384-d array for the record at row 42
  - `_id_list[42]` → `"vec_a1b2c3d4"` (SQLite record ID at row 42)
  - `_id_to_index["vec_a1b2c3d4"]` → `42` (reverse dict, O(1))
- Any mutation updates matrix row **and** ordered list **and** reverse dict before persisting
- Code reference: `core/vector_store.py`

### Slide 17 — Distance metrics from scratch (cosine, euclidean, dot)

- All three metrics implemented in `core/distance_metrics.py` with single-pair and batch versions
- Formulas and ranges:

| Metric | Formula | Range | Best |
|--------|---------|-------|------|
| Cosine similarity | `A·B / (‖A‖·‖B‖)` | `[-1, 1]` | `1.0` |
| Euclidean distance | `√(Σ(Aᵢ − Bᵢ)²)` | `[0, ∞)` | `0.0` |
| Dot product | `Σ(Aᵢ · Bᵢ)` | `(-∞, ∞)` | higher is better |

- **Batch fast path** (used by every search):

```python
# core/distance_metrics.py — batch_cosine_similarity (excerpt)
dot_products   = vectors @ query            # (N,384) @ (384,) → (N,)
vector_norms   = np.linalg.norm(vectors, axis=1)
query_norm     = np.linalg.norm(query)
denominators   = vector_norms * query_norm
similarities   = np.where(denominators != 0, dot_products / denominators, 0.0)
return np.clip(similarities, -1.0, 1.0)
```

- Strategy pattern: `METRIC_REGISTRY = {"cosine": {...}, "euclidean": {...}, "dot": {...}}` maps a name to `(single_fn, batch_fn, higher_is_better)` — adding a new metric = one dict entry

### Slide 18 — Embedding engine (neural + fallback)

- **`EmbeddingEngine`** (`core/embeddings.py`)
  - Model: `sentence-transformers/all-MiniLM-L6-v2` → 384-d `float32`
  - **Lazy loading**: model only downloads/loads on first `encode()` call
  - Cache folder: `db_run/model_cache/huggingface/` (project-local, ≈ 80 MB)
  - `encode(text)` for single texts, `encode_batch(texts, batch_size=32)` for bulk inserts
- **`SimpleEmbeddingEngine`** — bag-of-words fallback
  - No neural dependency — uses `hash(word) % dimension` placement + count + normalise
  - Dimension forced to 384 so the NumPy matrix layout stays compatible
- **Factory**: `create_embedding_engine(fallback=True)` auto-picks the neural engine if available, otherwise falls back with a warning
- Keeps CI / offline / first-boot runs functional without ML dependencies

### Slide 19 — INSERT data flow

```text
1. Input validation
   ├── text is non-empty string
   ├── collection exists
   └── ID not duplicate

2. Text → vector
   └── EmbeddingEngine.encode("Python is great")
       → np.array([0.12, -0.03, …], dtype=float32)  # (384,)

3. SQLite (atomic transaction, session-scoped)
   ├── INSERT INTO records  (id, session_id, collection_id, text, created_at)
   └── INSERT INTO metadata (record_id, key, value) × K

4. NumPy (in-memory)
   ├── _vectors = np.vstack([_vectors, new_vector])
   └── _id_list.append("vec_a1b2c3d4"); rebuild _id_to_index

5. Disk (atomic: write-to-tmp + os.replace)
   ├── np.save("vectors.npy", _vectors)
   └── json.dump(_id_list, "id_mapping.json")

6. Return "vec_a1b2c3d4"
```

- **Compensating action**: if `save()` fails after the SQLite commit, `VectorStore.insert()` rolls back the in-memory state **and** deletes the SQLite row so both stores stay consistent
- `insert_batch()` amortises model-encoding + SQLite transaction + disk save over N rows — 10–50× faster than calling `insert()` in a loop

### Slide 20 — SEARCH data flow

```text
1. Input validation (DB non-empty, metric name valid)

2. Embed the query text → query_vector shape (384,)

3. Pre-filter candidates (SQL)
   ├── filter_by_metadata(filters)  — INNER JOIN metadata × records on session_id
   ├── get_record_ids_in_collection(collection) if specified
   └── Convert matching record IDs → NumPy row indices via _id_to_index

4. Batch similarity (NumPy)
   ├── candidate_vectors = _vectors[filtered_indices]  # (M, 384)
   ├── scores = batch_fn(query_vector, candidate_vectors)  # (M,)
   └── batch_fn reduces to: candidate_vectors @ query_vector

5. Rank
   ├── sorted = np.argsort(scores)[::-1]   # descending (cosine/dot)
   └── or np.argsort(scores)               # ascending (euclidean)
   └── take top_k

6. Hydrate results
   └── For each top index: look up id from _id_list → fetch row + metadata from SQLite → SearchResult(record, score, rank, metric)
```

- Every user query is logged via `DatabaseManager.log_message(kind='search', query_text, metric, top_k, category_filter, result_count, elapsed_ms, response_ref)`
- The trigger `trg_touch_session_on_message` bumps the session's `last_used_at` on every insert to `messages`

### Slide 21 — File upload pipeline (TXT / CSV / Excel)

- Implemented in `core/file_processor.py`; shared by the web UI (`/upload`, `/insert`) and the CLI (`import-file`)
- **Validation**: extension in `{.txt, .csv, .xlsx, .xls}`, size ≤ **10 MB**
- **TXT path**: sentence/word chunking with configurable overlap
- **CSV / Excel path** (row-first, zero overlap between rows):
  - Detect likely header rows, skip blank noise, deduplicate repeated header-like rows
  - Serialise each logical row as a stable `"Column: value | Column: value | …"` string
  - Split an oversized row only within that row (never across rows)
- **Output**: `(texts, metadata_list)` — plugged directly into `VectorStore.insert_batch()`
- **Uses**: `csv` (stdlib), `pandas`, `openpyxl` — chosen for messy-header resilience

### Slide 22 — Command-line interface (10 subcommands)

- Entry point: `python -m cli.main <subcommand> [...]`
- **Subcommands** (argparse): `insert`, `search`, `get`, `update`, `delete`, `list`, `stats`, `collections`, `create-collection`, `import-file`
- **Global flags**: `--db-path`, `--new-run`, `--run-prefix`, `--model-cache-path`, `--session`
- Thin wrapper — parses args, calls `VectorStore`, prints formatted output
- If no active session exists and no flags are passed, the CLI lists available sessions and exits (no auto-seed)

**[SCREENSHOT #1]** — capture a terminal running: `python -m cli.main --new-run insert "Deep learning has revolutionised computer vision." --metadata category=science source=sample`

**[SCREENSHOT #2]** — capture: `python -m cli.main search "machine learning for medical imaging" --top-k 5 --metric cosine`

**[SCREENSHOT #3]** — capture: `python -m cli.main stats` showing `total_records`, `total_collections`, `memory_usage_bytes`, `embedding_model`

### Slide 23 — Web UI — session picker + search

- Flask app at `web/app.py`; Jinja2 templates at `web/templates/`
- **Routes**:
  - `GET /` → session picker (`select_session.html`)
  - `POST /session/new` → create run, rebind `VectorStore`, redirect
  - `POST /session/switch` → rebind to chosen session
  - `GET /search-page` → search form (redirects home if no active session)
  - `POST /search` → runs query, logs to `messages` with `kind='search'`
  - `GET /api/search` → JSON endpoint with the same logging path

**[SCREENSHOT #4]** — `http://localhost:5000/` showing the session picker (new session button + existing sessions table)

**[SCREENSHOT #5]** — `/search-page` showing the query field, metric selector (cosine / euclidean / dot), top-k slider, optional `filter_key` / `filter_value` fields

**[SCREENSHOT #6]** — `/search` (results view) showing ranked top-k results with scores, texts, and metadata badges

### Slide 24 — Web UI — insert, upload, history, stats, records

- **Routes**:
  - `GET/POST /insert` → two tabs: "Paste text" + "Upload file"
  - `GET/POST /upload` → file upload handler (TXT / CSV / Excel)
  - `GET /history` → chat-style timeline from `DatabaseManager.get_history()`
  - `GET /stats` → per-session aggregate stats (records, collections, memory, model)
  - `GET /records` → paginated record browser with collection filter
- Chat history stores **user queries only** — never per-result rows or large blobs; response metadata lives on the same `messages` row (`result_count`, `elapsed_ms`, `response_ref`)

**[SCREENSHOT #7]** — `/insert` with the "Paste text" tab active (text area + metadata key/value fields)

**[SCREENSHOT #8]** — `/insert` with the "Upload file" tab active (drag-and-drop or chooser, accepts `.txt/.csv/.xlsx/.xls`)

**[SCREENSHOT #9]** — `/history` timeline showing alternating `search` / `insert` messages with metric, top-k, result count, elapsed ms

**[SCREENSHOT #10]** — `/stats` page showing per-session aggregates and per-collection record counts

**[SCREENSHOT #11]** — `/records` paginated browser with collection filter and record rows

---

## 7. Project Results

### Slide 25 — Benchmark setup

- Driver: `tests/benchmark.py`; saved results: `tests/benchmark_results.json` (timestamp **2026-04-14T17:32:47**)
- **Corpus sizes**: 100, 500, 1000, 2000 records
- **Dimension**: 384
- **Workload**: 50 queries per run × 3 runs (averaged)
- **Metrics tested**: cosine, euclidean, dot (dedicated `bench_metric_comparison` phase)
- **Two engines compared**:
  - `simple_engine` — `SimpleEmbeddingEngine` (bag-of-words, no ML deps)
  - `real_engine` — `SentenceTransformer all-MiniLM-L6-v2` (model load ≈ 7.30 s on first use)
- Curated corpus at `data/generated/`: 5 JSON shards (technology, science, sports, health, business) — ≈ 151 docs used for semantic demo

### Slide 26 — Latency, metric ordering, memory footprint

- **BoW (SimpleBoW) — average query latency**
  - N = 1000 → ~0.39 ms
  - N = 2000 → ~0.84 ms
- **SentenceTransformer — average query latency (neural encoding dominates)**
  - N = 1000 → ~11.5 ms (overall)
  - Metric breakdown at N = 1000: cosine ≈ 11.2 ms, euclidean ≈ 12.0 ms, dot ≈ 11.2 ms
- **Qualitative metric ordering** (as expected by the math)
  - **dot** fastest (pure matrix–vector multiply)
  - **cosine** intermediate (adds norms + safe divide + clip)
  - **euclidean** slowest (broadcast subtract + squared sum + `sqrt`)
- **Memory (`store._vectors.nbytes`)**
  - 100 × 384 × 4 bytes ≈ **0.15 MB**
  - 2000 × 384 × 4 bytes ≈ **2.93 MB** (real measurement, not a hard-coded constant)
- **Complexity**: brute-force *k*-NN is **O(N·D)** per query after candidate selection — motivates ANN indexes (IVF, HNSW) as future work

### Slide 27 — Semantic search demo

- Driver: `demo/semantic_search.py` — 6-step end-to-end walkthrough
  1. Load the curated corpus (≈ 151 docs across 5 categories)
  2. Create a fresh `VectorStore(new_run=True)`
  3. Bulk insert via `insert_batch()`
  4. Run diverse natural-language queries; print top-k hits
  5. Apply a metadata filter (e.g. search within `category = science` only)
  6. Compare paraphrased query pairs; show overlap in retrieved IDs
- Example: the query **"machine learning for medical imaging"** returns both Health and Technology documents because transformer embeddings align on **meaning**, not term overlap
- Pre-filtering in SQL (e.g. `category = "Science"`) narrows the candidate set **before** the O(N·D) scan — inexpensive filters, dense fast path
- Chat history (`/history`) records the exact queries + metric + top-k + elapsed time for reproducibility

---

## 8. References

### Slide 28 — References

1. Reimers, N., and Gurevych, I. *"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."* Proceedings of EMNLP, 2019
2. Hugging Face Model Card — `sentence-transformers/all-MiniLM-L6-v2` — https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
3. NumPy Reference Documentation — https://numpy.org/doc/stable/
4. SQLite Documentation — CREATE TABLE / Triggers / Foreign Keys — https://sqlite.org/docs.html
5. Flask Documentation — https://flask.palletsprojects.com/
6. pandas Documentation — https://pandas.pydata.org/docs/
7. openpyxl Documentation — https://openpyxl.readthedocs.io/
8. Johnson, J., Douze, M., Jégou, H. *"Billion-scale similarity search with GPUs."* IEEE Transactions on Big Data, 2019 (FAISS)
9. Pinecone Vector Database Documentation — https://docs.pinecone.io/
10. ChromaDB Documentation — https://docs.trychroma.com/
11. Weaviate Documentation — https://weaviate.io/developers/weaviate
12. Codd, E. F. *"A Relational Model of Data for Large Shared Data Banks."* Communications of the ACM, 1970
13. pytest Documentation — https://docs.pytest.org/
14. MiniVecDB project source — `AGENTS.md`, `README.md`, `ARCHITECTURE.py`, and the full academic report at `.claude/PROJECT_REPORT.md`
