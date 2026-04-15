"""
MiniVecDB Performance Benchmark Suite.

This benchmark can run in two modes:
1. corpus mode (default): uses real documents from data/generated/*.json
2. synthetic mode: uses randomly generated text for stress testing

Both embedding engines are benchmarked:
- SimpleEmbeddingEngine (bag-of-words fallback)
- EmbeddingEngine (sentence-transformers)

Benchmarks:
1. Insertion throughput
2. Query latency (avg, p50, p95, min, max)
3. Vector memory usage (_vectors.nbytes)
4. Metric comparison (cosine, euclidean, dot)
"""

import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------
# Path setup: add project root to sys.path so we can import
# project modules the same way all other test files do.
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.embeddings import EmbeddingEngine, SimpleEmbeddingEngine
from core.runtime_paths import create_new_run_path
from core.vector_store import VectorStore
from data.sample_dataset import load_dataset


# ===============================================================
# CONFIGURATION
# ===============================================================


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


SIZES = [25, 50, 100, 151]              # Benchmark sizes tuned to our 151-doc corpus
QUERY_COUNT = 50                         # Queries per run
NUM_RUNS = 3                             # Repeat each measurement N times
METRICS = ["cosine", "euclidean", "dot"]
METRIC_COMPARISON_SIZE = 151             # Use full corpus for metric comparison
DIMENSION = 384                          # Embedding vector size

VALID_BENCHMARK_MODES = {"corpus", "synthetic"}
BENCHMARK_MODE = os.environ.get("MINIVECDB_BENCHMARK_MODE", "corpus").strip().lower()
if BENCHMARK_MODE not in VALID_BENCHMARK_MODES:
    print(
        f"[benchmark] Invalid MINIVECDB_BENCHMARK_MODE='{BENCHMARK_MODE}'. "
        "Falling back to 'corpus'."
    )
    BENCHMARK_MODE = "corpus"

RUN_PREFIX = os.environ.get("MINIVECDB_BENCH_RUN_PREFIX", "bench").strip() or "bench"
VERBOSE_QUERY_LOG = _env_flag("MINIVECDB_BENCH_VERBOSE_QUERIES", True)
PRESERVE_ARTIFACTS = _env_flag("MINIVECDB_BENCH_PRESERVE", True)

# Output path for JSON results
RESULTS_PATH = os.path.join(PROJECT_ROOT, "tests", "benchmark_results.json")

# Query extraction knobs for corpus mode
CORPUS_QUERY_SEED = 99
MIN_QUERY_WORDS = 6
MAX_QUERY_WORDS = 14

# Word pool used only in synthetic mode
WORD_POOL = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "algorithm", "database", "vector", "search", "neural", "network",
    "machine", "learning", "artificial", "intelligence", "python",
    "programming", "software", "engineering", "computer", "science",
    "quantum", "physics", "biology", "chemistry", "mathematics",
    "research", "analysis", "data", "model", "training", "inference",
    "performance", "benchmark", "latency", "throughput", "memory",
    "storage", "index", "query", "result", "score", "similarity",
    "distance", "metric", "cosine", "euclidean", "dimension",
    "embedding", "transformer", "attention", "layer", "token",
    "sentence", "paragraph", "document", "collection", "record",
    "insert", "delete", "update", "retrieve", "filter", "metadata",
    "category", "technology", "health", "sports", "business",
    "economy", "market", "startup", "innovation", "product",
    "customer", "strategy", "growth", "revenue", "profit",
    "exercise", "nutrition", "wellness", "mental", "physical",
    "football", "basketball", "cricket", "tennis", "olympics",
    "galaxy", "planet", "star", "universe", "energy", "atom",
    "molecule", "cell", "gene", "protein", "evolution", "species",
    "climate", "ocean", "mountain", "river", "forest", "desert",
    "city", "country", "population", "culture", "history", "future",
    "robot", "automation", "cloud", "server", "application", "system",
    "security", "encryption", "privacy", "blockchain", "digital",
    "signal", "frequency", "wave", "particle", "field", "force",
    "gravity", "relativity", "entropy", "thermodynamics", "catalyst",
    "reaction", "solution", "experiment", "hypothesis", "theory",
    "discovery", "invention", "patent", "journal", "conference",
    "professor", "student", "university", "laboratory", "telescope",
    "microscope", "satellite", "rocket", "mission", "exploration",
    "deep", "shallow", "complex", "simple", "abstract", "concrete",
    "global", "local", "dynamic", "static", "parallel", "sequential",
]

# Expected shard files (matches data/sample_dataset.py load order)
CORPUS_SHARD_FILES = [
    "technology.json",
    "science.json",
    "sports.json",
    "health.json",
    "business.json",
]


# ===============================================================
# TABLE + DISPLAY HELPERS
# ===============================================================


def format_table(title: str, headers: List[str], rows: List[List[str]]) -> str:
    """Format tabular data as a clean ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    col_widths = [w + 2 for w in col_widths]

    lines: List[str] = []
    lines.append(f"\n  {title}")
    lines.append("  " + "-" * (sum(col_widths) + len(col_widths) + 1))

    header_str = "  |"
    for h, w in zip(headers, col_widths):
        header_str += f" {h:<{w}}|"
    lines.append(header_str)

    sep_str = "  |"
    for w in col_widths:
        sep_str += "-" * (w + 1) + "|"
    lines.append(sep_str)

    for row in rows:
        row_str = "  |"
        for val, w in zip(row, col_widths):
            row_str += f" {val:<{w}}|"
        lines.append(row_str)

    lines.append("  " + "-" * (sum(col_widths) + len(col_widths) + 1))
    return "\n".join(lines)



def print_banner(text: str) -> None:
    """Print a prominent section banner."""
    width = 78
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_sub_banner(text: str) -> None:
    """Print a lighter sub-section banner for steps within a phase."""
    width = 70
    print("\n    " + "-" * width)
    print(f"    {text}")
    print("    " + "-" * width)


def print_step(label: str, message: str) -> None:
    """Print a consistently formatted step/status message."""
    print(f"      [{label}] {message}")


def print_detail(message: str) -> None:
    """Print an indented detail line under a step."""
    print(f"        {message}")


# ===============================================================
# TEXT GENERATION (SYNTHETIC MODE)
# ===============================================================


def generate_random_texts(n: int, seed: int = 42) -> List[str]:
    """Generate n random synthetic documents from WORD_POOL."""
    rng = random.Random(seed)
    texts: List[str] = []
    for _ in range(n):
        length = rng.randint(10, 30)
        words = rng.choices(WORD_POOL, k=length)
        texts.append(" ".join(words))
    return texts



def generate_query_texts(n: int, seed: int = 99) -> List[str]:
    """Generate n random synthetic query strings."""
    rng = random.Random(seed)
    queries: List[str] = []
    for _ in range(n):
        length = rng.randint(3, 8)
        words = rng.choices(WORD_POOL, k=length)
        queries.append(" ".join(words))
    return queries



def generate_metadata(n: int) -> List[Dict[str, Any]]:
    """Generate metadata dicts for synthetic documents."""
    categories = ["technology", "science", "sports", "health", "business"]
    subcategories = {
        "technology": ["ai", "programming", "gadgets", "software"],
        "science": ["physics", "biology", "chemistry", "astronomy"],
        "sports": ["football", "basketball", "cricket", "tennis"],
        "health": ["nutrition", "fitness", "mental_health", "medicine"],
        "business": ["startups", "finance", "marketing", "economy"],
    }
    metadata_list: List[Dict[str, Any]] = []
    for i in range(n):
        cat = categories[i % len(categories)]
        sub = subcategories[cat][i % len(subcategories[cat])]
        metadata_list.append(
            {
                "category": cat,
                "subcategory": sub,
                "source": "benchmark_synthetic",
            }
        )
    return metadata_list


# ===============================================================
# CORPUS MODE HELPERS
# ===============================================================


def normalize_collection_name(category: str) -> str:
    """Convert category text into a safe collection name."""
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", category.strip().lower())
    safe = safe.strip("_")
    return safe or "default"



def group_records_by_collection(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group normalized records by their target collection."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["collection"]].append(record)
    return {key: grouped[key] for key in sorted(grouped)}



def derive_corpus_sizes(total_docs: int, requested_sizes: List[int]) -> List[int]:
    """Pick benchmark sizes that do not exceed the real corpus size."""
    valid = sorted({size for size in requested_sizes if 0 < size <= total_docs})
    if total_docs not in valid:
        valid.append(total_docs)
    valid = sorted(set(valid))

    # If the corpus is small and only one size survives, include half-size too.
    if len(valid) == 1 and total_docs > 1:
        half_size = max(1, total_docs // 2)
        if half_size not in valid:
            valid.insert(0, half_size)

    return valid



def select_corpus_records(
    records_by_collection: Dict[str, List[Dict[str, Any]]],
    size: int,
) -> List[Dict[str, Any]]:
    """Select a deterministic, balanced subset via round-robin across collections."""
    selected: List[Dict[str, Any]] = []
    collections = sorted(records_by_collection)
    cursors = {collection: 0 for collection in collections}

    while len(selected) < size:
        progressed = False
        for collection in collections:
            docs = records_by_collection[collection]
            idx = cursors[collection]
            if idx < len(docs):
                selected.append(docs[idx])
                cursors[collection] += 1
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            break

    if len(selected) != size:
        raise ValueError(
            f"Unable to select {size} records from corpus. Selected {len(selected)} instead."
        )

    # Show selection distribution
    sel_grouped = defaultdict(int)
    for record in selected:
        sel_grouped[record["collection"]] += 1
    print_step("select", f"Selected {size} records via round-robin:")
    for collection in sorted(sel_grouped):
        print_detail(f"{collection}: {sel_grouped[collection]} docs")

    return selected



def _extract_query_phrase(text: str, rng: random.Random) -> str:
    """Extract a deterministic short phrase from a document for query generation."""
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    candidate = parts[rng.randrange(len(parts))] if parts else text.strip()

    words = candidate.split()
    if len(words) < MIN_QUERY_WORDS:
        words = text.split()

    if len(words) > MAX_QUERY_WORDS:
        max_start = max(0, len(words) - MAX_QUERY_WORDS)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        words = words[start : start + MAX_QUERY_WORDS]

    if len(words) < MIN_QUERY_WORDS:
        words = text.split()[:MIN_QUERY_WORDS]

    return " ".join(words).strip()



def generate_corpus_query_pool(
    records: List[Dict[str, Any]],
    seed: int = CORPUS_QUERY_SEED,
) -> List[Dict[str, Any]]:
    """Generate deterministic, corpus-valid queries from real document content.

    Every query is derived from an actual document in the corpus -- no random
    word salad.  For each document we extract a short phrase and optionally
    prefix it with the subcategory to create a second variant.
    """
    rng = random.Random(seed)
    query_pool: List[Dict[str, Any]] = []
    seen: set[str] = set()

    print_step("query-gen", f"Generating query pool from {len(records)} corpus documents (seed={seed})...")

    for idx, record in enumerate(records):
        phrase = _extract_query_phrase(record["text"], rng)
        if not phrase:
            continue

        collection = record["collection"]
        subcategory = str(record["metadata"].get("subcategory", "unknown")).strip()

        for candidate in (phrase, f"{subcategory} {phrase}".strip()):
            key = candidate.lower()
            if not candidate or key in seen:
                continue
            seen.add(key)
            query_pool.append(
                {
                    "query": candidate,
                    "collection": collection,
                    "subcategory": subcategory,
                    "source_index": idx,
                }
            )

    if not query_pool:
        raise ValueError("Failed to generate corpus query pool from dataset.")

    # Show per-collection breakdown of generated queries
    pool_by_collection: Dict[str, int] = defaultdict(int)
    for entry in query_pool:
        pool_by_collection[entry["collection"]] += 1
    print_step("query-gen", f"Query pool generated: {len(query_pool)} unique queries")
    for col in sorted(pool_by_collection):
        print_detail(f"{col}: {pool_by_collection[col]} queries")

    # Show a few sample queries with their source document
    print_step("query-gen", "Sample corpus-derived queries:")
    for sample in query_pool[:6]:
        src_text = records[sample["source_index"]]["text"]
        preview = src_text[:80] + "..." if len(src_text) > 80 else src_text
        print_detail(
            f"[doc #{sample['source_index']}] ({sample['collection']}/{sample['subcategory']})"
        )
        print_detail(f"  Source : {preview}")
        print_detail(f"  Query  : {sample['query']}")

    return query_pool



def build_corpus_query_workload(
    query_pool: List[Dict[str, Any]],
    query_count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build a fixed-size deterministic query workload from a corpus query pool."""
    if not query_pool:
        raise ValueError("query_pool must be non-empty.")

    rng = random.Random(seed)
    indices = list(range(len(query_pool)))
    rng.shuffle(indices)

    workload: List[Dict[str, Any]] = []
    cursor = 0
    while len(workload) < query_count:
        workload.append(query_pool[indices[cursor]])
        cursor += 1
        if cursor >= len(indices):
            cursor = 0

    # Show workload composition
    wl_by_collection: Dict[str, int] = defaultdict(int)
    for entry in workload:
        wl_by_collection[entry["collection"]] += 1
    print_step(
        "workload",
        f"Built query workload: {len(workload)} queries from pool of {len(query_pool)} (seed={seed})",
    )
    for col in sorted(wl_by_collection):
        print_detail(f"{col}: {wl_by_collection[col]} queries")

    return workload



def load_corpus_records() -> List[Dict[str, Any]]:
    """Load and validate corpus records from data/generated/*.json."""
    print_banner("CORPUS LOAD")
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "generated")
    print(f"  [corpus] Dataset directory: {dataset_dir}")

    # Show each shard file before loading
    print(f"  [corpus] Expected shard files ({len(CORPUS_SHARD_FILES)}):")
    for shard_name in CORPUS_SHARD_FILES:
        shard_path = os.path.join(dataset_dir, shard_name)
        exists = os.path.isfile(shard_path)
        size_kb = os.path.getsize(shard_path) / 1024 if exists else 0
        status = f"{size_kb:.1f} KB" if exists else "MISSING"
        print(f"    - {shard_name:<20} {status}")

    print(f"\n  [corpus] Calling load_dataset() from data/sample_dataset.py ...")
    dataset = load_dataset()
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset is empty or invalid. Expected a non-empty list of records.")

    print(f"  [corpus] Raw dataset loaded: {len(dataset)} records")

    # Validate and normalize every record
    normalized_records: List[Dict[str, Any]] = []
    category_counter: Dict[str, int] = defaultdict(int)
    subcategory_counter: Dict[str, int] = defaultdict(int)

    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item {idx} is not a dictionary.")

        text = item.get("text", "")
        metadata = item.get("metadata", {})
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Dataset item {idx} has empty or invalid 'text'.")
        if not isinstance(metadata, dict):
            raise ValueError(f"Dataset item {idx} has invalid 'metadata'.")

        category_raw = str(metadata.get("category", "default")).strip() or "default"
        collection = normalize_collection_name(category_raw)

        normalized_metadata: Dict[str, str] = {}
        for key, value in metadata.items():
            key_str = str(key).strip()
            if key_str:
                normalized_metadata[key_str] = str(value)

        normalized_metadata["category"] = category_raw
        normalized_metadata["benchmark_collection"] = collection

        normalized_records.append(
            {
                "text": text.strip(),
                "metadata": normalized_metadata,
                "collection": collection,
            }
        )
        category_counter[category_raw] += 1
        subcategory_counter[str(metadata.get("subcategory", "unknown"))] += 1

    # Show per-category and per-subcategory breakdown
    grouped = group_records_by_collection(normalized_records)
    rows = [[collection, str(len(docs))] for collection, docs in grouped.items()]
    print(format_table("CORPUS COLLECTION BREAKDOWN", ["Collection", "Documents"], rows))

    sub_rows = [[sub, str(cnt)] for sub, cnt in sorted(subcategory_counter.items())]
    print(format_table("CORPUS SUBCATEGORY BREAKDOWN", ["Subcategory", "Documents"], sub_rows))

    # Show a sample record from each collection
    print("\n  [corpus] Sample record from each collection:")
    for collection, docs in grouped.items():
        first = docs[0]
        text_preview = first["text"][:90] + "..." if len(first["text"]) > 90 else first["text"]
        sub = first["metadata"].get("subcategory", "?")
        print(f"    [{collection}] (sub={sub}) \"{text_preview}\"")

    print(f"\n  [corpus] Total records loaded and validated: {len(normalized_records)}")
    return normalized_records



def prepare_corpus_context(requested_sizes: List[int]) -> Dict[str, Any]:
    """Build reusable corpus context for corpus-mode benchmarking."""
    records = load_corpus_records()
    records_by_collection = group_records_by_collection(records)
    total_records = len(records)

    print_banner("CORPUS CONTEXT PREPARATION")
    print(f"  [context] Total corpus documents: {total_records}")
    print(f"  [context] Requested benchmark sizes: {requested_sizes}")

    effective_sizes = derive_corpus_sizes(total_records, requested_sizes)
    print(f"  [context] Effective sizes after clamping to corpus: {effective_sizes}")

    if set(effective_sizes) != set(requested_sizes):
        dropped = sorted(set(requested_sizes) - set(effective_sizes))
        if dropped:
            print(f"  [context] Sizes dropped (exceed {total_records}-doc corpus): {dropped}")

    print(f"\n  [context] Generating corpus-derived query pool...")
    query_pool = generate_corpus_query_pool(records, seed=CORPUS_QUERY_SEED)

    print(f"\n  [context] Full query pool: {len(query_pool)} unique queries")
    print(f"  [context] Sample generated queries:")
    for sample in query_pool[:8]:
        print(
            f"    - ({sample['collection']}/{sample['subcategory']}) "
            f"\"{sample['query'][:70]}{'...' if len(sample['query']) > 70 else ''}\""
        )

    return {
        "records": records,
        "records_by_collection": records_by_collection,
        "total_records": total_records,
        "sizes": effective_sizes,
        "query_pool": query_pool,
        "collection_counts": {
            collection: len(docs) for collection, docs in records_by_collection.items()
        },
    }


# ===============================================================
# STORE + EXECUTION HELPERS
# ===============================================================


def _engine_label(use_simple: bool) -> str:
    """Return short engine label used in logs and run-name prefixes."""
    return "simple" if use_simple else "transformer"



def _create_benchmark_storage(stage: str, use_simple: bool, suffix: str) -> str:
    """Create a managed run directory under db_run for benchmark artifacts."""
    prefix = f"{RUN_PREFIX}_{BENCHMARK_MODE}_{stage}_{_engine_label(use_simple)}_{suffix}"
    storage_dir = create_new_run_path(prefix=prefix)
    print_step("storage", f"Created run directory: {storage_dir}")
    return storage_dir



def _make_store(storage_dir: str, use_simple: bool = True) -> VectorStore:
    """Create a VectorStore for one benchmark run."""
    print_step("store", f"Initializing VectorStore...")
    print_detail(f"Storage path  : {storage_dir}")
    print_detail(f"SQLite DB     : {os.path.join(storage_dir, 'minivecdb.db')}")
    print_detail(f"Vectors file  : {os.path.join(storage_dir, 'vectors.npy')}")
    print_detail(f"ID mapping    : {os.path.join(storage_dir, 'id_mapping.json')}")

    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=DIMENSION,
    )

    if use_simple:
        store.embedding_engine = SimpleEmbeddingEngine(dimension=DIMENSION)
        print_detail(f"Engine        : SimpleEmbeddingEngine (bag-of-words, dim={DIMENSION})")
    else:
        model_name = getattr(store.embedding_engine, "model_name", "unknown")
        print_detail(f"Engine        : EmbeddingEngine (model={model_name}, dim={DIMENSION})")

    return store



def _ensure_collections(store: VectorStore, collection_names: List[str]) -> None:
    """Ensure required collections exist before collection-specific batch inserts."""
    unique = sorted(set(collection_names))
    print_step("collections", f"Ensuring {len(unique)} collections exist in SQLite...")

    for collection in unique:
        if collection == store.DEFAULT_COLLECTION:
            print_detail(f"'{collection}' (default) -- already exists")
            continue
        try:
            store.create_collection(
                name=collection,
                description=f"Benchmark collection '{collection}'",
            )
            print_detail(f"'{collection}' -- created (dim={DIMENSION})")
        except ValueError:
            print_detail(f"'{collection}' -- already exists")



def _insert_records_grouped(
    store: VectorStore,
    grouped_records: Dict[str, List[Dict[str, Any]]],
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """Insert grouped records collection-by-collection with detailed logs."""
    total_docs = sum(len(docs) for docs in grouped_records.values())
    print_step(
        "insert",
        f"Inserting {total_docs} documents across {len(grouped_records)} collections...",
    )

    inserted_total = 0
    per_collection: Dict[str, Dict[str, Any]] = {}

    for collection, records in grouped_records.items():
        texts = [record["text"] for record in records]
        metadata_list = [record["metadata"] for record in records]
        print_step(
            "insert",
            f"Collection '{collection}': {len(texts)} documents "
            f"(running total: {inserted_total}/{total_docs})",
        )

        start = time.perf_counter()
        store.insert_batch(texts=texts, metadata_list=metadata_list, collection=collection)
        elapsed = time.perf_counter() - start

        inserted_total += len(texts)
        docs_per_sec = round(len(texts) / elapsed, 1) if elapsed > 0 else 0.0
        per_collection[collection] = {
            "count": len(texts),
            "time_s": round(elapsed, 4),
            "docs_per_sec": docs_per_sec,
        }
        print_detail(
            f"Done in {elapsed:.4f}s ({docs_per_sec:.1f} docs/sec) -- "
            f"cumulative: {inserted_total}/{total_docs}"
        )

    print_step("insert", f"All insertions complete: {inserted_total} documents total")
    print_detail(f"Vector matrix shape: {store._vectors.shape}")
    print_detail(f"Vector memory: {store._vectors.nbytes:,} bytes ({store._vectors.nbytes / 1024:.1f} KB)")

    return inserted_total, per_collection



def _run_query_workload(
    store: VectorStore,
    workload: List[Dict[str, Any]],
    metric: str,
) -> Tuple[List[float], int]:
    """Execute a query workload and return latency list plus non-empty result count."""

    # Show workload summary before running
    wl_collections: Dict[str, int] = defaultdict(int)
    for item in workload:
        wl_collections[item.get("collection") or "all"] += 1
    print_step(
        "query",
        f"Running {len(workload)} queries (metric={metric}, top_k=5)...",
    )
    for col in sorted(wl_collections):
        print_detail(f"Collection '{col}': {wl_collections[col]} queries")

    latencies_ms: List[float] = []
    non_empty_results = 0

    for idx, item in enumerate(workload, 1):
        query = item["query"]
        collection = item.get("collection")

        start = time.perf_counter()
        results = store.search(
            query=query,
            top_k=5,
            metric=metric,
            collection=collection,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        if results:
            non_empty_results += 1

        if VERBOSE_QUERY_LOG:
            collection_label = collection or "all"
            top_score = f"{results[0].score:.4f}" if results else "N/A"
            top_text = ""
            if results:
                top_text = results[0].record.text[:50] + "..." if len(results[0].record.text) > 50 else results[0].record.text
            print(
                f"      [query {idx:03d}] metric={metric:<9} "
                f"collection={collection_label:<12} latency={elapsed_ms:.4f}ms "
                f"results={len(results)} top_score={top_score}"
            )
            if top_text:
                print(f"                   top_match: \"{top_text}\"")

    # Show query phase summary
    arr = np.array(latencies_ms, dtype=np.float64)
    print_step(
        "query",
        f"Workload complete: {len(workload)} queries, "
        f"{non_empty_results} returned results ({non_empty_results / len(workload) * 100:.0f}%)",
    )
    print_detail(
        f"Latency: avg={np.mean(arr):.4f}ms, "
        f"p50={np.percentile(arr, 50):.4f}ms, "
        f"p95={np.percentile(arr, 95):.4f}ms"
    )

    return latencies_ms, non_empty_results



def _latency_summary(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute aggregate latency stats."""
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "avg_ms": round(float(np.mean(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
    }



def _make_synthetic_grouped_records(size: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """Create synthetic records grouped by collection."""
    texts = generate_random_texts(size, seed=seed)
    metadata_list = generate_metadata(size)

    records: List[Dict[str, Any]] = []
    for text, metadata in zip(texts, metadata_list):
        collection = normalize_collection_name(str(metadata.get("category", "default")))
        records.append(
            {
                "text": text,
                "metadata": metadata,
                "collection": collection,
            }
        )

    return group_records_by_collection(records)



def _build_synthetic_query_workload(query_count: int, seed: int = 99) -> List[Dict[str, Any]]:
    """Build synthetic query workload with deterministic collection assignment."""
    base_queries = generate_query_texts(query_count, seed=seed)
    collections = ["technology", "science", "sports", "health", "business"]

    workload: List[Dict[str, Any]] = []
    for idx, query in enumerate(base_queries):
        workload.append(
            {
                "query": query,
                "collection": collections[idx % len(collections)],
            }
        )
    return workload


# ===============================================================
# MODEL AVAILABILITY + PREWARM
# ===============================================================


def is_real_model_available() -> bool:
    """Check if sentence-transformers is importable."""
    engine = EmbeddingEngine()
    available = engine._check_availability()
    print(f"  [engine-check] sentence-transformers available: {available}")
    return available



def prewarm_real_model() -> Dict[str, Any]:
    """Load the real model once and return load time with artifact path."""
    print_sub_banner("MODEL PREWARM")
    storage_dir = _create_benchmark_storage("prewarm", use_simple=False, suffix="model")
    store: Optional[VectorStore] = None

    try:
        store = _make_store(storage_dir, use_simple=False)
        print_step("prewarm", "Triggering first encode() to warm model into memory...")
        start = time.perf_counter()
        store.embedding_engine.encode("warmup sentence for model loading")
        elapsed = time.perf_counter() - start
        print_step("prewarm", f"Model warmup complete in {elapsed:.4f}s")
        return {
            "model_load_time_s": round(elapsed, 4),
            "storage_path": storage_dir,
        }
    finally:
        if store is not None:
            store.close()

        # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
        # shutil.rmtree(storage_dir, ignore_errors=True)
        if PRESERVE_ARTIFACTS:
            print_step("artifact", f"Preserved prewarm artifacts at: {storage_dir}")


# ===============================================================
# BENCHMARK 1: INSERTION THROUGHPUT
# ===============================================================


def _bench_insertion_synthetic(sizes: List[int], use_simple: bool) -> List[Dict[str, Any]]:
    """Insertion throughput benchmark for synthetic mode."""
    results: List[Dict[str, Any]] = []

    for size in sizes:
        run_times: List[float] = []
        run_paths: List[str] = []

        for run_idx in range(NUM_RUNS):
            print_sub_banner(
                f"INSERT (synthetic) | size={size} | run {run_idx + 1}/{NUM_RUNS} | "
                f"engine={_engine_label(use_simple)}"
            )
            grouped_records = _make_synthetic_grouped_records(size=size, seed=42 + run_idx)
            storage_dir = _create_benchmark_storage(
                stage="insert",
                use_simple=use_simple,
                suffix=f"n{size}_r{run_idx + 1}",
            )

            store: Optional[VectorStore] = None
            try:
                store = _make_store(storage_dir, use_simple=use_simple)
                _ensure_collections(store, list(grouped_records.keys()))

                start = time.perf_counter()
                inserted_count, breakdown = _insert_records_grouped(store, grouped_records)
                elapsed = time.perf_counter() - start

                run_times.append(elapsed)
                run_paths.append(storage_dir)

                print_step(
                    "result",
                    f"Run {run_idx + 1} complete: {inserted_count} docs in {elapsed:.4f}s "
                    f"({size / elapsed:.1f} docs/sec)",
                )
            finally:
                if store is not None:
                    store.close()

                # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
                # shutil.rmtree(storage_dir, ignore_errors=True)
                if PRESERVE_ARTIFACTS:
                    print_step("artifact", f"Preserved: {storage_dir}")

        avg_time = sum(run_times) / len(run_times)
        docs_per_sec = size / avg_time if avg_time > 0 else 0.0
        results.append(
            {
                "size": size,
                "avg_time_s": round(avg_time, 4),
                "docs_per_sec": round(docs_per_sec, 1),
                "run_times_s": [round(value, 4) for value in run_times],
                "run_paths": run_paths,
            }
        )

    return results



def _bench_insertion_corpus(
    sizes: List[int],
    use_simple: bool,
    corpus_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Insertion throughput benchmark for corpus mode."""
    results: List[Dict[str, Any]] = []
    corpus_grouped = corpus_context["records_by_collection"]

    for size in sizes:
        run_times: List[float] = []
        run_paths: List[str] = []

        for run_idx in range(NUM_RUNS):
            print_sub_banner(
                f"INSERT (corpus) | size={size} | run {run_idx + 1}/{NUM_RUNS} | "
                f"engine={_engine_label(use_simple)}"
            )
            selected = select_corpus_records(corpus_grouped, size=size)
            grouped_records = group_records_by_collection(selected)
            storage_dir = _create_benchmark_storage(
                stage="insert",
                use_simple=use_simple,
                suffix=f"n{size}_r{run_idx + 1}",
            )

            store: Optional[VectorStore] = None
            try:
                store = _make_store(storage_dir, use_simple=use_simple)
                _ensure_collections(store, list(grouped_records.keys()))

                start = time.perf_counter()
                inserted_count, breakdown = _insert_records_grouped(store, grouped_records)
                elapsed = time.perf_counter() - start

                run_times.append(elapsed)
                run_paths.append(storage_dir)

                print_step(
                    "result",
                    f"Run {run_idx + 1} complete: {inserted_count} docs in {elapsed:.4f}s "
                    f"({size / elapsed:.1f} docs/sec)",
                )
            finally:
                if store is not None:
                    store.close()

                # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
                # shutil.rmtree(storage_dir, ignore_errors=True)
                if PRESERVE_ARTIFACTS:
                    print_step("artifact", f"Preserved: {storage_dir}")

        avg_time = sum(run_times) / len(run_times)
        docs_per_sec = size / avg_time if avg_time > 0 else 0.0
        results.append(
            {
                "size": size,
                "avg_time_s": round(avg_time, 4),
                "docs_per_sec": round(docs_per_sec, 1),
                "run_times_s": [round(value, 4) for value in run_times],
                "run_paths": run_paths,
            }
        )

    return results



def bench_insertion(
    sizes: List[int],
    use_simple: bool = True,
    corpus_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Dispatch insertion benchmark to synthetic or corpus implementation."""
    if BENCHMARK_MODE == "corpus":
        if corpus_context is None:
            raise ValueError("corpus_context is required when BENCHMARK_MODE='corpus'.")
        return _bench_insertion_corpus(sizes=sizes, use_simple=use_simple, corpus_context=corpus_context)
    return _bench_insertion_synthetic(sizes=sizes, use_simple=use_simple)


# ===============================================================
# BENCHMARK 2: QUERY LATENCY
# ===============================================================


def _bench_query_latency_synthetic(sizes: List[int], use_simple: bool) -> List[Dict[str, Any]]:
    """Query latency benchmark for synthetic mode."""
    results: List[Dict[str, Any]] = []

    for size in sizes:
        all_latencies_ms: List[float] = []
        non_empty_total = 0
        run_paths: List[str] = []

        for run_idx in range(NUM_RUNS):
            print_sub_banner(
                f"QUERY LATENCY (synthetic) | size={size} | run {run_idx + 1}/{NUM_RUNS} | "
                f"engine={_engine_label(use_simple)}"
            )
            grouped_records = _make_synthetic_grouped_records(size=size, seed=42 + run_idx)
            workload = _build_synthetic_query_workload(query_count=QUERY_COUNT, seed=99 + run_idx)
            storage_dir = _create_benchmark_storage(
                stage="query",
                use_simple=use_simple,
                suffix=f"n{size}_r{run_idx + 1}",
            )

            store: Optional[VectorStore] = None
            try:
                store = _make_store(storage_dir, use_simple=use_simple)
                _ensure_collections(store, list(grouped_records.keys()))
                _insert_records_grouped(store, grouped_records)

                latencies, non_empty = _run_query_workload(
                    store=store,
                    workload=workload,
                    metric="cosine",
                )
                all_latencies_ms.extend(latencies)
                non_empty_total += non_empty
                run_paths.append(storage_dir)
            finally:
                if store is not None:
                    store.close()

                # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
                # shutil.rmtree(storage_dir, ignore_errors=True)
                if PRESERVE_ARTIFACTS:
                    print_step("artifact", f"Preserved: {storage_dir}")

        summary = _latency_summary(all_latencies_ms)
        total_queries = len(all_latencies_ms)
        results.append(
            {
                "size": size,
                "total_queries": total_queries,
                "non_empty_ratio": round(non_empty_total / total_queries, 4) if total_queries else 0.0,
                "run_paths": run_paths,
                **summary,
            }
        )

    return results



def _bench_query_latency_corpus(
    sizes: List[int],
    use_simple: bool,
    corpus_context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Query latency benchmark for corpus mode with corpus-derived queries."""
    results: List[Dict[str, Any]] = []
    corpus_grouped = corpus_context["records_by_collection"]

    for size in sizes:
        all_latencies_ms: List[float] = []
        non_empty_total = 0
        run_paths: List[str] = []

        for run_idx in range(NUM_RUNS):
            print_sub_banner(
                f"QUERY LATENCY (corpus) | size={size} | run {run_idx + 1}/{NUM_RUNS} | "
                f"engine={_engine_label(use_simple)}"
            )
            selected = select_corpus_records(corpus_grouped, size=size)
            grouped_records = group_records_by_collection(selected)

            query_pool = generate_corpus_query_pool(selected, seed=CORPUS_QUERY_SEED + run_idx)
            workload = build_corpus_query_workload(
                query_pool=query_pool,
                query_count=QUERY_COUNT,
                seed=CORPUS_QUERY_SEED + 1000 + run_idx,
            )

            storage_dir = _create_benchmark_storage(
                stage="query",
                use_simple=use_simple,
                suffix=f"n{size}_r{run_idx + 1}",
            )

            store: Optional[VectorStore] = None
            try:
                store = _make_store(storage_dir, use_simple=use_simple)
                _ensure_collections(store, list(grouped_records.keys()))
                _insert_records_grouped(store, grouped_records)

                print_step("query", "Running corpus-derived query workload against real documents...")
                latencies, non_empty = _run_query_workload(
                    store=store,
                    workload=workload,
                    metric="cosine",
                )

                all_latencies_ms.extend(latencies)
                non_empty_total += non_empty
                run_paths.append(storage_dir)
            finally:
                if store is not None:
                    store.close()

                # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
                # shutil.rmtree(storage_dir, ignore_errors=True)
                if PRESERVE_ARTIFACTS:
                    print_step("artifact", f"Preserved: {storage_dir}")

        summary = _latency_summary(all_latencies_ms)
        total_queries = len(all_latencies_ms)
        results.append(
            {
                "size": size,
                "total_queries": total_queries,
                "non_empty_ratio": round(non_empty_total / total_queries, 4) if total_queries else 0.0,
                "run_paths": run_paths,
                **summary,
            }
        )

    return results



def bench_query_latency(
    sizes: List[int],
    use_simple: bool = True,
    corpus_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Dispatch query-latency benchmark by selected mode."""
    if BENCHMARK_MODE == "corpus":
        if corpus_context is None:
            raise ValueError("corpus_context is required when BENCHMARK_MODE='corpus'.")
        return _bench_query_latency_corpus(sizes=sizes, use_simple=use_simple, corpus_context=corpus_context)
    return _bench_query_latency_synthetic(sizes=sizes, use_simple=use_simple)


# ===============================================================
# BENCHMARK 3: MEMORY USAGE
# ===============================================================


def _bench_memory_synthetic(sizes: List[int]) -> List[Dict[str, Any]]:
    """Memory usage benchmark for synthetic mode."""
    results: List[Dict[str, Any]] = []

    for size in sizes:
        print_sub_banner(f"MEMORY (synthetic) | size={size}")
        grouped_records = _make_synthetic_grouped_records(size=size, seed=42)
        storage_dir = _create_benchmark_storage(
            stage="memory",
            use_simple=True,
            suffix=f"n{size}",
        )

        store: Optional[VectorStore] = None
        try:
            store = _make_store(storage_dir, use_simple=True)
            _ensure_collections(store, list(grouped_records.keys()))
            inserted_count, _ = _insert_records_grouped(store, grouped_records)

            nbytes = store._vectors.nbytes
            shape = store._vectors.shape
            results.append(
                {
                    "size": inserted_count,
                    "shape": list(shape),
                    "bytes": nbytes,
                    "kb": round(nbytes / 1024, 2),
                    "mb": round(nbytes / (1024 * 1024), 4),
                    "storage_path": storage_dir,
                }
            )
            print_step("memory", f"shape={shape}, {nbytes:,} bytes, {nbytes / (1024 * 1024):.4f} MB")
            print_detail(f"Per-vector: {nbytes // shape[0]:,} bytes ({DIMENSION} x 4 bytes = float32)")
        finally:
            if store is not None:
                store.close()

            # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
            # shutil.rmtree(storage_dir, ignore_errors=True)
            if PRESERVE_ARTIFACTS:
                print_step("artifact", f"Preserved: {storage_dir}")

    return results



def _bench_memory_corpus(sizes: List[int], corpus_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Memory usage benchmark for corpus mode."""
    results: List[Dict[str, Any]] = []
    corpus_grouped = corpus_context["records_by_collection"]

    for size in sizes:
        print_sub_banner(f"MEMORY (corpus) | size={size}")
        selected = select_corpus_records(corpus_grouped, size=size)
        grouped_records = group_records_by_collection(selected)
        storage_dir = _create_benchmark_storage(
            stage="memory",
            use_simple=True,
            suffix=f"n{size}",
        )

        store: Optional[VectorStore] = None
        try:
            store = _make_store(storage_dir, use_simple=True)
            _ensure_collections(store, list(grouped_records.keys()))
            inserted_count, _ = _insert_records_grouped(store, grouped_records)

            nbytes = store._vectors.nbytes
            shape = store._vectors.shape
            results.append(
                {
                    "size": inserted_count,
                    "shape": list(shape),
                    "bytes": nbytes,
                    "kb": round(nbytes / 1024, 2),
                    "mb": round(nbytes / (1024 * 1024), 4),
                    "storage_path": storage_dir,
                }
            )
            print_step("memory", f"shape={shape}, {nbytes:,} bytes, {nbytes / (1024 * 1024):.4f} MB")
            print_detail(f"Per-vector: {nbytes // shape[0]:,} bytes ({DIMENSION} x 4 bytes = float32)")
            print_detail(f"Vectors file on disk: {os.path.join(storage_dir, 'vectors.npy')}")
        finally:
            if store is not None:
                store.close()

            # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
            # shutil.rmtree(storage_dir, ignore_errors=True)
            if PRESERVE_ARTIFACTS:
                print_step("artifact", f"Preserved: {storage_dir}")

    return results



def bench_memory(
    sizes: List[int],
    corpus_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Dispatch memory benchmark by selected mode."""
    if BENCHMARK_MODE == "corpus":
        if corpus_context is None:
            raise ValueError("corpus_context is required when BENCHMARK_MODE='corpus'.")
        return _bench_memory_corpus(sizes=sizes, corpus_context=corpus_context)
    return _bench_memory_synthetic(sizes=sizes)


# ===============================================================
# BENCHMARK 4: METRIC COMPARISON
# ===============================================================


def _bench_metric_comparison_synthetic(use_simple: bool = True) -> Dict[str, Any]:
    """Metric comparison benchmark for synthetic mode."""
    size = METRIC_COMPARISON_SIZE
    metric_results: Dict[str, Any] = {}
    run_paths_by_metric: Dict[str, List[str]] = {}

    for metric_name in METRICS:
        print_sub_banner(
            f"METRIC COMPARISON (synthetic) | metric={metric_name} | size={size} | "
            f"engine={_engine_label(use_simple)}"
        )
        all_latencies_ms: List[float] = []
        non_empty_total = 0
        run_paths: List[str] = []

        for run_idx in range(NUM_RUNS):
            grouped_records = _make_synthetic_grouped_records(size=size, seed=42 + run_idx)
            workload = _build_synthetic_query_workload(query_count=QUERY_COUNT, seed=99 + run_idx)
            storage_dir = _create_benchmark_storage(
                stage=f"metric_{metric_name}",
                use_simple=use_simple,
                suffix=f"n{size}_r{run_idx + 1}",
            )

            store: Optional[VectorStore] = None
            try:
                store = _make_store(storage_dir, use_simple=use_simple)
                _ensure_collections(store, list(grouped_records.keys()))
                _insert_records_grouped(store, grouped_records)

                latencies, non_empty = _run_query_workload(
                    store=store,
                    workload=workload,
                    metric=metric_name,
                )
                all_latencies_ms.extend(latencies)
                non_empty_total += non_empty
                run_paths.append(storage_dir)
            finally:
                if store is not None:
                    store.close()

                # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
                # shutil.rmtree(storage_dir, ignore_errors=True)
                if PRESERVE_ARTIFACTS:
                    print_step("artifact", f"Preserved: {storage_dir}")

        summary = _latency_summary(all_latencies_ms)
        total_queries = len(all_latencies_ms)
        metric_results[metric_name] = {
            "total_queries": total_queries,
            "non_empty_ratio": round(non_empty_total / total_queries, 4) if total_queries else 0.0,
            **summary,
        }
        run_paths_by_metric[metric_name] = run_paths

    return {
        "size": size,
        "metrics": metric_results,
        "run_paths_by_metric": run_paths_by_metric,
    }



def _bench_metric_comparison_corpus(
    use_simple: bool,
    corpus_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Metric comparison benchmark for corpus mode."""
    size = min(METRIC_COMPARISON_SIZE, corpus_context["total_records"])
    selected = select_corpus_records(corpus_context["records_by_collection"], size=size)
    grouped_records = group_records_by_collection(selected)

    metric_results: Dict[str, Any] = {}
    run_paths_by_metric: Dict[str, List[str]] = {}

    for metric_name in METRICS:
        print_sub_banner(
            f"METRIC COMPARISON (corpus) | metric={metric_name} | size={size} | "
            f"engine={_engine_label(use_simple)}"
        )
        all_latencies_ms: List[float] = []
        non_empty_total = 0
        run_paths: List[str] = []

        for run_idx in range(NUM_RUNS):
            query_pool = generate_corpus_query_pool(selected, seed=CORPUS_QUERY_SEED + run_idx)
            workload = build_corpus_query_workload(
                query_pool=query_pool,
                query_count=QUERY_COUNT,
                seed=CORPUS_QUERY_SEED + 2000 + run_idx,
            )

            storage_dir = _create_benchmark_storage(
                stage=f"metric_{metric_name}",
                use_simple=use_simple,
                suffix=f"n{size}_r{run_idx + 1}",
            )

            store: Optional[VectorStore] = None
            try:
                store = _make_store(storage_dir, use_simple=use_simple)
                _ensure_collections(store, list(grouped_records.keys()))
                _insert_records_grouped(store, grouped_records)

                latencies, non_empty = _run_query_workload(
                    store=store,
                    workload=workload,
                    metric=metric_name,
                )
                all_latencies_ms.extend(latencies)
                non_empty_total += non_empty
                run_paths.append(storage_dir)
            finally:
                if store is not None:
                    store.close()

                # Cleanup intentionally disabled to preserve benchmark artifacts for inspection.
                # shutil.rmtree(storage_dir, ignore_errors=True)
                if PRESERVE_ARTIFACTS:
                    print_step("artifact", f"Preserved: {storage_dir}")

        summary = _latency_summary(all_latencies_ms)
        total_queries = len(all_latencies_ms)
        metric_results[metric_name] = {
            "total_queries": total_queries,
            "non_empty_ratio": round(non_empty_total / total_queries, 4) if total_queries else 0.0,
            **summary,
        }
        run_paths_by_metric[metric_name] = run_paths

    return {
        "size": size,
        "metrics": metric_results,
        "run_paths_by_metric": run_paths_by_metric,
    }



def bench_metric_comparison(
    use_simple: bool = True,
    corpus_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dispatch metric benchmark by selected mode."""
    if BENCHMARK_MODE == "corpus":
        if corpus_context is None:
            raise ValueError("corpus_context is required when BENCHMARK_MODE='corpus'.")
        return _bench_metric_comparison_corpus(use_simple=use_simple, corpus_context=corpus_context)
    return _bench_metric_comparison_synthetic(use_simple=use_simple)


# ===============================================================
# ORCHESTRATOR
# ===============================================================


def run_all_benchmarks() -> Dict[str, Any]:
    """Run the complete benchmark suite for selected mode and both engines."""
    suite_start = time.perf_counter()

    print_banner("BENCHMARK CONFIGURATION")
    print(f"  Mode              : {BENCHMARK_MODE}")
    print(f"  Requested sizes   : {SIZES}")
    print(f"  Query count/run   : {QUERY_COUNT}")
    print(f"  Runs per benchmark: {NUM_RUNS}")
    print(f"  Metric set        : {METRICS}")
    print(f"  Dimension         : {DIMENSION}")
    print(f"  Run prefix        : {RUN_PREFIX}")
    print(f"  Preserve artifacts: {PRESERVE_ARTIFACTS}")
    print(f"  Verbose queries   : {VERBOSE_QUERY_LOG}")
    print(f"  Results output    : {RESULTS_PATH}")

    corpus_context: Optional[Dict[str, Any]] = None
    effective_sizes = list(SIZES)

    if BENCHMARK_MODE == "corpus":
        corpus_context = prepare_corpus_context(requested_sizes=SIZES)
        effective_sizes = corpus_context["sizes"]

    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": BENCHMARK_MODE,
            "requested_sizes": SIZES,
            "effective_sizes": effective_sizes,
            "query_count": QUERY_COUNT,
            "num_runs": NUM_RUNS,
            "dimension": DIMENSION,
            "metrics": METRICS,
            "run_prefix": RUN_PREFIX,
            "preserve_artifacts": PRESERVE_ARTIFACTS,
            "verbose_query_logging": VERBOSE_QUERY_LOG,
        },
    }

    if corpus_context is not None:
        results["corpus"] = {
            "total_records": corpus_context["total_records"],
            "collection_counts": corpus_context["collection_counts"],
            "query_pool_size": len(corpus_context["query_pool"]),
        }

    # ----------------------------------------------------------
    # Simple engine benchmarks
    # ----------------------------------------------------------
    print_banner("BENCHMARKING: SimpleBoW Engine")
    simple_start = time.perf_counter()

    print("\n  [1/4] Insertion throughput...")
    simple_insertion = bench_insertion(
        sizes=effective_sizes,
        use_simple=True,
        corpus_context=corpus_context,
    )
    print(
        format_table(
            "INSERTION THROUGHPUT -- SimpleBoW Engine",
            ["N", "Avg Time (s)", "Docs/sec"],
            [
                [
                    str(row["size"]),
                    f"{row['avg_time_s']:.4f}",
                    f"{row['docs_per_sec']:.1f}",
                ]
                for row in simple_insertion
            ],
        )
    )

    print("\n  [2/4] Query latency...")
    simple_query = bench_query_latency(
        sizes=effective_sizes,
        use_simple=True,
        corpus_context=corpus_context,
    )
    print(
        format_table(
            f"QUERY LATENCY -- SimpleBoW Engine ({QUERY_COUNT} queries x {NUM_RUNS} runs)",
            ["N", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Min (ms)", "Max (ms)", "Non-empty"],
            [
                [
                    str(row["size"]),
                    f"{row['avg_ms']:.4f}",
                    f"{row['p50_ms']:.4f}",
                    f"{row['p95_ms']:.4f}",
                    f"{row['min_ms']:.4f}",
                    f"{row['max_ms']:.4f}",
                    f"{row['non_empty_ratio']:.2f}",
                ]
                for row in simple_query
            ],
        )
    )

    print("\n  [3/4] Memory usage...")
    memory = bench_memory(sizes=effective_sizes, corpus_context=corpus_context)
    print(
        format_table(
            "MEMORY USAGE -- _vectors.nbytes",
            ["N", "Shape", "Bytes", "KB", "MB"],
            [
                [
                    str(row["size"]),
                    str(row["shape"]),
                    f"{row['bytes']:,}",
                    f"{row['kb']:.2f}",
                    f"{row['mb']:.4f}",
                ]
                for row in memory
            ],
        )
    )

    print("\n  [4/4] Metric comparison...")
    simple_metrics = bench_metric_comparison(use_simple=True, corpus_context=corpus_context)
    print(
        format_table(
            f"METRIC COMPARISON at N={simple_metrics['size']} -- SimpleBoW Engine",
            ["Metric", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Non-empty"],
            [
                [
                    name,
                    f"{data['avg_ms']:.4f}",
                    f"{data['p50_ms']:.4f}",
                    f"{data['p95_ms']:.4f}",
                    f"{data['non_empty_ratio']:.2f}",
                ]
                for name, data in simple_metrics["metrics"].items()
            ],
        )
    )

    simple_elapsed = time.perf_counter() - simple_start
    print(f"\n  SimpleBoW phase complete in {simple_elapsed:.2f}s")

    results["simple_engine"] = {
        "name": "SimpleBoW (bag-of-words fallback)",
        "phase_time_s": round(simple_elapsed, 2),
        "insertion": simple_insertion,
        "query_latency": simple_query,
        "metric_comparison": simple_metrics,
    }
    results["memory_usage"] = memory

    # ----------------------------------------------------------
    # Real SentenceTransformer benchmarks
    # ----------------------------------------------------------
    real_model_ok = is_real_model_available()

    if not real_model_ok:
        print("\n" + "!" * 78)
        print("  WARNING: sentence-transformers not installed.")
        print("  Skipping real model benchmarks.")
        print("  Install with: pip install sentence-transformers")
        print("!" * 78)
        results["real_engine"] = {
            "name": "SentenceTransformer (not available)",
            "skipped": True,
        }
    else:
        print_banner("BENCHMARKING: SentenceTransformer Engine (all-MiniLM-L6-v2)")
        real_start = time.perf_counter()

        try:
            print("\n  Pre-warming model (cache hit or download will be reported)...")
            prewarm_info = prewarm_real_model()
            print(f"  Model warmed in {prewarm_info['model_load_time_s']:.4f}s")
        except Exception as exc:
            print("\n" + "!" * 78)
            print("  WARNING: Failed to prewarm/load real model.")
            print(f"  Reason: {type(exc).__name__}: {exc}")
            print("  Skipping real model benchmarks for this run.")
            print("!" * 78)
            results["real_engine"] = {
                "name": "SentenceTransformer (failed to initialize)",
                "skipped": True,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            real_model_ok = False
        else:
            print("\n  [1/3] Insertion throughput...")
            real_insertion = bench_insertion(
                sizes=effective_sizes,
                use_simple=False,
                corpus_context=corpus_context,
            )
            print(
                format_table(
                    "INSERTION THROUGHPUT -- SentenceTransformer Engine",
                    ["N", "Avg Time (s)", "Docs/sec"],
                    [
                        [
                            str(row["size"]),
                            f"{row['avg_time_s']:.4f}",
                            f"{row['docs_per_sec']:.1f}",
                        ]
                        for row in real_insertion
                    ],
                )
            )

            print("\n  [2/3] Query latency...")
            real_query = bench_query_latency(
                sizes=effective_sizes,
                use_simple=False,
                corpus_context=corpus_context,
            )
            print(
                format_table(
                    f"QUERY LATENCY -- SentenceTransformer Engine ({QUERY_COUNT} queries x {NUM_RUNS} runs)",
                    ["N", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Min (ms)", "Max (ms)", "Non-empty"],
                    [
                        [
                            str(row["size"]),
                            f"{row['avg_ms']:.4f}",
                            f"{row['p50_ms']:.4f}",
                            f"{row['p95_ms']:.4f}",
                            f"{row['min_ms']:.4f}",
                            f"{row['max_ms']:.4f}",
                            f"{row['non_empty_ratio']:.2f}",
                        ]
                        for row in real_query
                    ],
                )
            )

            print("\n  [3/3] Metric comparison...")
            real_metrics = bench_metric_comparison(use_simple=False, corpus_context=corpus_context)
            print(
                format_table(
                    f"METRIC COMPARISON at N={real_metrics['size']} -- SentenceTransformer Engine",
                    ["Metric", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Non-empty"],
                    [
                        [
                            name,
                            f"{data['avg_ms']:.4f}",
                            f"{data['p50_ms']:.4f}",
                            f"{data['p95_ms']:.4f}",
                            f"{data['non_empty_ratio']:.2f}",
                        ]
                        for name, data in real_metrics["metrics"].items()
                    ],
                )
            )

            real_elapsed = time.perf_counter() - real_start
            print(f"\n  SentenceTransformer phase complete in {real_elapsed:.2f}s")

            results["real_engine"] = {
                "name": "SentenceTransformer (all-MiniLM-L6-v2)",
                "phase_time_s": round(real_elapsed, 2),
                "model_load_time_s": prewarm_info["model_load_time_s"],
                "prewarm_storage_path": prewarm_info["storage_path"],
                "insertion": real_insertion,
                "query_latency": real_query,
                "metric_comparison": real_metrics,
            }

    if real_model_ok:
        print_banner("SIDE-BY-SIDE COMPARISON")
        print_comparison_tables(results)

    # Grand summary
    suite_elapsed = time.perf_counter() - suite_start
    print_banner("GRAND SUMMARY")
    print(f"  Mode                : {BENCHMARK_MODE}")
    print(f"  Effective sizes     : {effective_sizes}")
    if corpus_context:
        print(f"  Corpus documents    : {corpus_context['total_records']}")
        print(f"  Collections         : {list(corpus_context['collection_counts'].keys())}")
        print(f"  Query pool size     : {len(corpus_context['query_pool'])}")
    print(f"  Queries per size    : {QUERY_COUNT} x {NUM_RUNS} runs = {QUERY_COUNT * NUM_RUNS} total")
    print(f"  SimpleBoW tested    : Yes")
    print(f"  Transformer tested  : {'Yes' if real_model_ok else 'No (skipped)'}")
    print(f"  Total suite time    : {suite_elapsed:.2f}s")
    print(f"  Artifacts preserved : {PRESERVE_ARTIFACTS}")

    return results



def print_comparison_tables(results: Dict[str, Any]) -> None:
    """Print side-by-side throughput and query-latency comparison."""
    simple_ins = results["simple_engine"]["insertion"]
    real_ins = results["real_engine"]["insertion"]

    print(
        format_table(
            "INSERTION THROUGHPUT -- Side by Side",
            ["N", "SimpleBoW (doc/s)", "Transformer (doc/s)", "Speedup"],
            [
                [
                    str(simple["size"]),
                    f"{simple['docs_per_sec']:.1f}",
                    f"{real['docs_per_sec']:.1f}",
                    (
                        f"{simple['docs_per_sec'] / real['docs_per_sec']:.1f}x"
                        if real["docs_per_sec"] > 0
                        else "N/A"
                    ),
                ]
                for simple, real in zip(simple_ins, real_ins)
            ],
        )
    )

    simple_ql = results["simple_engine"]["query_latency"]
    real_ql = results["real_engine"]["query_latency"]

    print(
        format_table(
            "QUERY LATENCY (avg ms) -- Side by Side",
            ["N", "SimpleBoW (ms)", "Transformer (ms)", "Ratio"],
            [
                [
                    str(simple["size"]),
                    f"{simple['avg_ms']:.4f}",
                    f"{real['avg_ms']:.4f}",
                    f"{real['avg_ms'] / simple['avg_ms']:.1f}x" if simple["avg_ms"] > 0 else "N/A",
                ]
                for simple, real in zip(simple_ql, real_ql)
            ],
        )
    )


# ===============================================================
# SAVE RESULTS
# ===============================================================


def save_results(results: Dict[str, Any], path: str = RESULTS_PATH) -> str:
    """Save benchmark results as JSON."""
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=2)
    return path


# ===============================================================
# MAIN
# ===============================================================


def main() -> None:
    """Entry point for benchmark suite."""
    print("\n" + "+" * 78)
    print("+  MiniVecDB -- Performance Benchmark Suite")
    print("+")
    print(f"+  Mode              : {BENCHMARK_MODE}")
    print(f"+  Requested sizes   : {SIZES}")
    print(f"+  Query count/run   : {QUERY_COUNT}")
    print(f"+  Runs per benchmark: {NUM_RUNS}")
    print(f"+  Engines           : SimpleBoW + SentenceTransformer")
    print(f"+  Metrics           : {', '.join(METRICS)}")
    print(f"+  Dimension         : {DIMENSION}")
    print(f"+  Run prefix        : {RUN_PREFIX}")
    print(f"+  Preserve artifacts: {PRESERVE_ARTIFACTS}")
    print(f"+  Results output    : {RESULTS_PATH}")
    print("+" * 78)

    total_start = time.perf_counter()

    results = run_all_benchmarks()

    json_path = save_results(results)
    total_elapsed = time.perf_counter() - total_start

    print_banner("BENCHMARK COMPLETE")
    print(f"  Total wall-clock time: {total_elapsed:.2f}s")
    print(f"  Results saved to     : {json_path}")
    print(f"  Results file size    : {os.path.getsize(json_path) / 1024:.1f} KB")
    print()


if __name__ == "__main__":
    main()
