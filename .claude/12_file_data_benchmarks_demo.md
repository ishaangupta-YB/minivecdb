# MiniVecDB — File: data/ (Dataset Module) & tests/benchmark.py (Performance Benchmarks)

> **Last updated**: Benchmark suite overhauled — corpus mode, realistic sizes, rich monitoring output, cache detection

---

## Part 1: The `data/` Module — Curated Sample Dataset

### Overview

The `data/` directory contains a **curated dataset of 151 documents** across 5 real-world categories, used by both the semantic search demo and the benchmark suite.

```
data/
├── __init__.py            # Package marker
├── sample_dataset.py      # Dataset loader — reads all 5 shards, returns unified list
└── generated/             # 5 JSON shard files (151 docs total)
    ├── technology.json     # 30 docs — AI, Programming, Gadgets, Software
    ├── science.json        # 31 docs — Physics, Biology, Chemistry, Astronomy (space)
    ├── sports.json         # 30 docs — Football, Basketball, Cricket, Olympics
    ├── health.json         # 30 docs — Nutrition, Exercise, Mental Health, Medicine
    └── business.json       # 30 docs — Startups, Finance, Marketing, Economy
```

**Exact corpus size**: 151 documents (technology: 30, science: 31, sports: 30, health: 30, business: 30).

---

### File: `data/sample_dataset.py`

#### Function: `load_dataset() → list`

Reads all 5 JSON shards in a fixed order and concatenates them into a single list. If a shard is missing, it warns and continues.

```python
def load_dataset():
    dataset_dir = Path(__file__).parent / "generated"
    shard_files = ["technology.json", "science.json", "sports.json", "health.json", "business.json"]
    all_records = []
    for shard_file in shard_files:
        shard_path = dataset_dir / shard_file
        if shard_path.exists():
            with open(shard_path, "r") as f:
                shard_data = json.load(f)
                all_records.extend(shard_data)
        else:
            print(f"Warning: {shard_file} not found at {shard_path}")
    return all_records
```

**Returns**: A flat list of dicts, each with `"text"` and `"metadata"` keys:
```python
{
    "text": "Lionel Messi led Argentina to victory in the 2022 FIFA World Cup...",
    "metadata": {
        "category": "Sports",
        "subcategory": "football",
        "source": "sample"
    }
}
```

---

### Dataset Structure

Each shard is a JSON array. Every record has exactly three metadata fields: `category`, `subcategory`, `source`.

#### Category → Subcategory Breakdown

| Category | Subcategories | Docs | File |
|---------|---------------|------|------|
| Technology | AI, Programming, Gadgets, Software | 30 | `technology.json` |
| Science | physics, biology, chemistry, space | 31 | `science.json` |
| Sports | football, cricket, basketball, Olympics | 30 | `sports.json` |
| Health | nutrition, exercise, mental health, medicine | 30 | `health.json` |
| Business | startups, finance, marketing, economy | 30 | `business.json` |
| **Total** | **20 subcategories** | **151** | |

**Note on capitalization**: `category` values use title case (`"Sports"`, `"Technology"`) while `subcategory` values are lowercase (`"football"`, `"ai"`). The benchmark normalizes categories into safe collection names via `normalize_collection_name()` (e.g. `"Sports"` → `"sports"`).

---

## Part 2: `tests/benchmark.py` — Performance Benchmark Suite

### Overview

> **Location**: `minivecdb/tests/benchmark.py`
> **Lines**: ~1,800 | **Purpose**: Systematically measures MiniVecDB's performance for insertion, search, memory, and metric comparison using BOTH embedding engines against the REAL corpus

The benchmark runs in two modes:
- **corpus mode** (default): uses the 151 real documents from `data/generated/*.json`
- **synthetic mode**: generates random word-salad texts for stress testing

Both modes test both engines:
- `SimpleEmbeddingEngine` (bag-of-words, no dependencies)
- `EmbeddingEngine` (sentence-transformers `all-MiniLM-L6-v2`)

---

### Configuration Constants

```python
SIZES = [25, 50, 100, 151]          # Benchmark sizes tuned to our 151-doc corpus
QUERY_COUNT = 50                     # Queries per run
NUM_RUNS = 3                         # Repeats for averaging
METRICS = ["cosine", "euclidean", "dot"]
METRIC_COMPARISON_SIZE = 151         # Use full corpus for metric comparison
DIMENSION = 384                      # Vector dimensionality

BENCHMARK_MODE = "corpus"            # Controlled by MINIVECDB_BENCHMARK_MODE env var
RUN_PREFIX = "bench"                 # Prefix for db_run artifact folders
PRESERVE_ARTIFACTS = True            # Keep db_run/ folders after benchmark
VERBOSE_QUERY_LOG = True             # Print per-query latency + top match text
```

**Why `[25, 50, 100, 151]`?** All four sizes fit within the real 151-document corpus. The old sizes `[100, 500, 1000, 2000]` had three sizes that exceeded the corpus and were silently clamped, leaving only two usable data points.

**Why `METRIC_COMPARISON_SIZE = 151`?** The old value of 1000 exceeded the corpus and was clamped anyway — making it explicit removes confusion.

---

### How Corpus Documents Load

```
main()
  └─► run_all_benchmarks()
        └─► prepare_corpus_context()
              └─► load_corpus_records()
                    └─► load_dataset()  [from data/sample_dataset.py]
                          └─► reads 5 JSON shards from data/generated/
                                → technology.json (30)
                                → science.json   (31)
                                → sports.json    (30)
                                → health.json    (30)
                                → business.json  (30)
                          → returns 151 raw records
                    → normalizes: category → collection name, coerces all metadata to str
                    → groups by collection
              → derive_corpus_sizes(): clamps [25, 50, 100, 151] against 151 → all survive
              → generate_corpus_query_pool(): extracts real phrases from real documents
```

---

### Query Generation — Corpus-Derived, Not Random

In corpus mode, every search query is a **real phrase extracted from an actual document**. There is no random word salad.

#### `generate_corpus_query_pool(records, seed) → List[Dict]`

For each document:
1. Splits the text on sentence boundaries (`.`, `!`, `?`)
2. Picks a random sentence using a seeded RNG (deterministic)
3. Extracts a 6–14 word phrase window from that sentence
4. Emits two query variants: the phrase alone, and `"{subcategory} {phrase}"`
5. Deduplicates across the pool using a `seen` set

Each query entry carries `query`, `collection`, `subcategory`, and `source_index` (the index of the document it came from). This means every query is guaranteed to have at least one semantically relevant result in the database.

#### `build_corpus_query_workload(query_pool, query_count, seed) → List[Dict]`

Shuffles the pool deterministically and samples exactly `QUERY_COUNT` (50) queries from it, cycling if the pool is smaller than the workload size.

**Print output** during query generation shows:
- Number of corpus documents used
- Per-collection query count breakdown
- 6 sample queries with their source document index, source text preview, and the derived query string

---

### Corpus Record Selection — Round-Robin for Balance

#### `select_corpus_records(records_by_collection, size) → List[Dict]`

Uses a round-robin cursor across all collections to ensure each collection is represented proportionally in every benchmark run. For size=25 with 5 collections of ~30 docs each: each collection contributes 5 docs.

Prints the per-collection distribution after every selection.

---

### VectorStore Setup Per Run

Each benchmark run creates a **fresh isolated VectorStore** under `db_run/`:

```
db_run/
└── bench_corpus_insert_simple_n25_r1_<timestamp>_<hex>/
    ├── minivecdb.db        ← SQLite: records, metadata, collections tables
    ├── vectors.npy         ← NumPy (N, 384) float32 matrix
    └── id_mapping.json     ← row index → record ID bridge
```

#### `_make_store(storage_dir, use_simple) → VectorStore`

Creates a `VectorStore` at the given path, then optionally replaces the embedding engine:
- `use_simple=True` → swaps to `SimpleEmbeddingEngine(dimension=384)` (bag-of-words)
- `use_simple=False` → keeps the default `EmbeddingEngine` (sentence-transformers)

Prints: storage path, SQLite DB path, vectors.npy path, id_mapping.json path, engine name.

#### `_ensure_collections(store, collection_names)`

Calls `store.create_collection()` for each unique collection name except `"default"` (which already exists). Catches `ValueError` for already-existing collections gracefully.

Prints each collection created or already-existing.

#### `_insert_records_grouped(store, grouped_records)`

Iterates over collections, calls `store.insert_batch()` for each one, times each batch individually, and prints:
- Running total (e.g. `25/100` docs inserted so far)
- Per-collection docs/sec
- Final vector matrix shape and memory usage in bytes/KB

---

### Benchmark #1: Insertion Throughput

#### `bench_insertion(sizes, use_simple, corpus_context) → List[Dict]`

For each size N in `[25, 50, 100, 151]`:
- Repeat `NUM_RUNS` (3) times:
  - Select N balanced corpus documents via round-robin
  - Create a fresh `VectorStore` run directory
  - Ensure all collections exist
  - Time the full grouped insert (all collections, all docs)
- Average the run times, compute docs/sec

**Result format**:
```json
{"size": 100, "avg_time_s": 0.0048, "docs_per_sec": 20833.3, "run_times_s": [0.0049, 0.0048, 0.0047], "run_paths": [...]}
```

**Synthetic vs corpus dispatch**: `bench_insertion()` is a dispatcher that calls `_bench_insertion_corpus()` or `_bench_insertion_synthetic()` based on `BENCHMARK_MODE`.

---

### Benchmark #2: Query Latency

#### `bench_query_latency(sizes, use_simple, corpus_context) → List[Dict]`

For each size N:
- Repeat `NUM_RUNS` (3) times:
  - Insert N corpus documents into a fresh store
  - Generate a corpus-derived query pool from those exact N documents
  - Build a 50-query workload from the pool
  - Run all 50 queries, timing each one individually in milliseconds
- Aggregate: 150 total latency samples (3 runs × 50 queries)
- Compute avg, p50, p95, min, max using NumPy

**Verbose query log** (enabled by default) prints per-query:
- Query index, metric, collection, latency in ms, result count, top score
- Top match text preview (first 50 characters)

**Result format**:
```json
{"size": 100, "total_queries": 150, "non_empty_ratio": 1.0, "avg_ms": 0.21, "p50_ms": 0.20, "p95_ms": 0.28, "min_ms": 0.17, "max_ms": 0.40}
```

**`non_empty_ratio`**: Fraction of queries that returned at least one result. In corpus mode this should always be 1.0 since every query is derived from a document in the store.

---

### Benchmark #3: Memory Usage

#### `bench_memory(sizes, corpus_context) → List[Dict]`

For each size N: inserts N corpus documents (using `SimpleEmbeddingEngine` only, since memory is engine-independent), then reads `store._vectors.nbytes`.

**Formula**: `bytes = N × 384 × 4` (384 dimensions × 4 bytes per float32)

Prints:
- Vector matrix shape
- Bytes, KB, MB
- Per-vector byte size (always 1,536 bytes = 384 × 4)
- Full path to the `vectors.npy` artifact on disk

**Result format**:
```json
{"size": 151, "shape": [151, 384], "bytes": 232,064, "kb": 226.62, "mb": 0.2213}
```

---

### Benchmark #4: Metric Comparison

#### `bench_metric_comparison(use_simple, corpus_context) → Dict`

At `METRIC_COMPARISON_SIZE = 151` (full corpus), runs the same 50-query workload three times — once per distance metric: `cosine`, `euclidean`, `dot`.

Measures query latency for each metric to show computational cost differences:
- **Dot product**: matrix multiply only — fastest
- **Cosine**: dot product + two norm computations + division
- **Euclidean**: subtraction + squared sum + square root

---

### Orchestrator: `run_all_benchmarks() → Dict`

Runs the full suite in phases, tracking elapsed time per phase:

```
Corpus context preparation (load 151 docs, generate query pool)

Phase 1: SimpleBoW Engine
  [1/4] Insertion throughput — sizes [25, 50, 100, 151]
  [2/4] Query latency — sizes [25, 50, 100, 151]
  [3/4] Memory usage — sizes [25, 50, 100, 151]
  [4/4] Metric comparison — N=151, all 3 metrics

Phase 2: SentenceTransformer Engine (if available)
  [Pre-warm] Load model, measure load time (cache vs download)
  [1/3] Insertion throughput
  [2/3] Query latency
  [3/3] Metric comparison

Phase 3: Side-by-side comparison tables
  Insertion speedup: SimpleBoW docs/sec ÷ Transformer docs/sec
  Latency ratio: Transformer avg_ms ÷ SimpleBoW avg_ms

Grand Summary
  Mode, effective sizes, corpus stats, query pool size, total queries, phase times
```

---

### Display Helpers

Three helper functions give consistent output formatting:

| Function | Indentation | Purpose |
|----------|-------------|---------|
| `print_banner(text)` | none | Full-width `===` banner for major phases |
| `print_sub_banner(text)` | 4 spaces | Lighter `---` separator within a phase |
| `print_step(label, message)` | 6 spaces | `[label] message` — individual operation status |
| `print_detail(message)` | 8 spaces | Supporting detail under a step |

---

### Supporting Functions

| Function | Purpose |
|----------|---------|
| `load_corpus_records()` | Loads all 151 docs, validates, normalizes metadata, groups by collection |
| `prepare_corpus_context()` | Builds reusable context: records, grouped, sizes, query pool |
| `normalize_collection_name(cat)` | `"Sports"` → `"sports"`, safe regex substitution |
| `group_records_by_collection(records)` | Groups a flat record list into a dict keyed by collection |
| `derive_corpus_sizes(total, requested)` | Clamps requested sizes to corpus size, adds full-corpus size if missing |
| `select_corpus_records(grouped, size)` | Round-robin balanced selection across collections |
| `_extract_query_phrase(text, rng)` | Extracts a 6–14 word phrase from a sentence in the text |
| `generate_corpus_query_pool(records, seed)` | Builds full pool of corpus-derived queries with source tracking |
| `build_corpus_query_workload(pool, n, seed)` | Shuffles pool, picks n queries deterministically |
| `_make_store(dir, use_simple)` | Creates VectorStore, swaps engine if needed, prints paths |
| `_ensure_collections(store, names)` | Creates required collections in SQLite |
| `_insert_records_grouped(store, grouped)` | Inserts collection-by-collection with timing and progress |
| `_run_query_workload(store, workload, metric)` | Runs all queries, times each, prints verbose log |
| `_latency_summary(latencies_ms)` | Computes avg, p50, p95, min, max via NumPy |
| `_create_benchmark_storage(stage, engine, suffix)` | Creates a named `db_run/` folder for this run |
| `is_real_model_available()` | Checks if sentence-transformers is importable |
| `prewarm_real_model()` | Loads model once to warm cache; returns load time |
| `format_table(title, headers, rows)` | Custom ASCII table formatter |
| `print_comparison_tables(results)` | Side-by-side insertion + latency comparison tables |
| `save_results(results, path)` | Writes full results dict to `tests/benchmark_results.json` |

---

### Artifacts Preserved After Each Run

Cleanup is **intentionally disabled** (commented-out `shutil.rmtree` lines throughout). Every benchmark run leaves its full VectorStore on disk under `db_run/` for inspection:

```
db_run/
├── bench_corpus_insert_simple_n25_r1_<ts>_<hex>/
│   ├── minivecdb.db       ← can be opened with sqlite3 to inspect records
│   ├── vectors.npy        ← can be loaded with np.load() to inspect embeddings
│   └── id_mapping.json    ← list mapping row index → record ID
├── bench_corpus_insert_simple_n50_r1_<ts>_<hex>/
│   └── ...
...
```

Each run directory name encodes: `{prefix}_{mode}_{stage}_{engine}_{size}_{run}_{timestamp}_{hex}`.

---

### Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `MINIVECDB_BENCHMARK_MODE` | `corpus` | `corpus` or `synthetic` |
| `MINIVECDB_BENCH_RUN_PREFIX` | `bench` | Prefix for `db_run/` folder names |
| `MINIVECDB_BENCH_PRESERVE` | `true` | Keep artifact folders after each run |
| `MINIVECDB_BENCH_VERBOSE_QUERIES` | `true` | Print per-query latency and top match |

---

### Running the Benchmarks

```bash
# Default: corpus mode, both engines
python tests/benchmark.py

# Synthetic mode (random texts, faster, no real data)
MINIVECDB_BENCHMARK_MODE=synthetic python tests/benchmark.py

# Quiet query log
MINIVECDB_BENCH_VERBOSE_QUERIES=false python tests/benchmark.py

# Results saved to
tests/benchmark_results.json
```

**Expected runtime**:
- SimpleBoW only: ~5–10 seconds (instant embeddings)
- Both engines: ~2–5 minutes per size with the SentenceTransformer model (most time is neural encoding)

---

## Part 3: `core/embeddings.py` — Cache vs Download Detection

A small but important enhancement was added to `EmbeddingEngine._load_model()`.

### New Method: `_detect_cached_model(cache_folder) → bool`

Before loading `SentenceTransformer`, the engine now checks whether the model files already exist locally. HuggingFace stores downloaded models using one of two naming conventions depending on the library version:
- Older: `sentence-transformers_all-MiniLM-L6-v2/`
- Newer: `models--sentence-transformers--all-MiniLM-L6-v2/`

The method scans `cache_folder` for directories matching either pattern and returns `True` if found.

### Print Output in `_load_model()`

**If model is in cache:**
```
[embeddings] Model 'all-MiniLM-L6-v2' found in local cache.
[embeddings] Loading from cache: /path/to/db_run/model_cache/huggingface
[embeddings] (No download required -- using previously cached files)
[embeddings] Model loaded successfully (cache) in 1.23s
[embeddings] Output dimension: 384
[embeddings] Cache folder: /path/to/db_run/model_cache/huggingface
```

**If model needs to be downloaded:**
```
[embeddings] Model 'all-MiniLM-L6-v2' NOT found in cache.
[embeddings] Downloading model from HuggingFace (~80 MB)...
[embeddings] Download destination: /path/to/db_run/model_cache/huggingface
[embeddings] (This is a one-time download; future runs will use the cache)
[embeddings] Model loaded successfully (download) in 18.47s
[embeddings] Output dimension: 384
[embeddings] Cache folder: /path/to/db_run/model_cache/huggingface
```

The load time is measured with `time.time()` and includes both download (if needed) and the model initialization itself.

---

## Part 4: `demo/semantic_search.py` — End-to-End Demo

> **Location**: `minivecdb/demo/semantic_search.py`
> **Purpose**: A complete end-to-end demonstration showing MiniVecDB's capabilities

### What It Does

A 6-step demo that loads the real dataset, populates a VectorStore, and demonstrates various search capabilities:

| Step | Function | What It Shows |
|------|----------|---------------|
| 1 | `load_and_summarize_dataset()` | Loads 151 docs from `data/`, prints category breakdown |
| 2 | `create_store_and_insert(dataset)` | Creates fresh VectorStore with `new_run=True`, bulk-inserts all docs, prints timing |
| 3 | `run_example_queries(store)` | Runs 10 diverse queries (2 per category), shows top-3 results each |
| 4 | `run_filtered_search(store)` | Demonstrates metadata pre-filtering: searches only Science category |
| 5 | `print_database_stats(store)` | Shows total records, memory, model info, storage path |
| 6 | `demonstrate_semantic_similarity(store)` | Tests 3 pairs of same-meaning different-wording queries, calculates result overlap |

### Semantic Similarity Demo (Step 6)

Tests three pairs of semantically equivalent queries:

| Pair | Query A | Query B |
|------|---------|---------|
| 1 | "How does machine learning work?" | "Explain the fundamentals of AI and neural networks" |
| 2 | "Tips for staying healthy and fit" | "What should I do to improve my physical wellness?" |
| 3 | "How to start a new business?" | "What are the steps for launching a startup company?" |

For each pair: runs both queries independently, computes set intersection of result IDs, reports overlap percentage. High overlap proves semantic search works.

### Running

```bash
python demo/semantic_search.py
```

---

## Import Graph

```
data/sample_dataset.py
  └─► load_dataset()
        ├─► tests/benchmark.py  (load_corpus_records → prepare_corpus_context)
        └─► demo/semantic_search.py  (load_and_summarize_dataset)

core/vector_store.py  (VectorStore)
  ├─► tests/benchmark.py  (_make_store, insert, search)
  └─► demo/semantic_search.py  (create_store_and_insert, queries)

core/embeddings.py  (EmbeddingEngine + SimpleEmbeddingEngine)
  └─► tests/benchmark.py  (prewarm, is_real_model_available, engine swap)
```
