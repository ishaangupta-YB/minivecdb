"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Performance Benchmark Suite                      ║
║  File: minivecdb/tests/benchmark.py                           ║
║                                                               ║
║  Measures performance of the MiniVecDB vector database using  ║
║  BOTH embedding engines:                                      ║
║    A) SimpleEmbeddingEngine  (bag-of-words, hash-based)       ║
║    B) EmbeddingEngine        (sentence-transformers model)    ║
║                                                               ║
║  Benchmarks:                                                  ║
║    1. Insertion throughput    (docs/sec at N=100..2000)        ║
║    2. Query latency           (ms/query with p50, p95)        ║
║    3. Memory usage            (vector matrix bytes)           ║
║    4. Metric comparison       (cosine vs euclidean vs dot)    ║
║                                                               ║
║  Run:  python tests/benchmark.py                              ║
╚═══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import random
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

# ---------------------------------------------------------------
# Path setup: add project root to sys.path so we can import
# minivecdb modules the same way all other test files do.
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.vector_store import VectorStore
from core.embeddings import SimpleEmbeddingEngine, EmbeddingEngine


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SIZES = [100, 500, 1000, 2000]       # Database sizes to benchmark
QUERY_COUNT = 50                      # Number of queries per run
NUM_RUNS = 3                          # Repeat each measurement N times
METRICS = ["cosine", "euclidean", "dot"]
METRIC_COMPARISON_SIZE = 1000         # Fixed size for metric comparison
DIMENSION = 384                       # Vector dimensionality

# Output path for JSON results
RESULTS_PATH = os.path.join(PROJECT_ROOT, "tests", "benchmark_results.json")

# ═══════════════════════════════════════════════════════════════
# WORD POOL — used to generate random text documents
# ═══════════════════════════════════════════════════════════════
# A mix of common English words from different domains so the
# SimpleEmbeddingEngine (hash-based) produces varied vectors.

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


# ═══════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_random_texts(n: int, seed: int = 42) -> List[str]:
    """
    Generate n random documents (10-30 words each) from WORD_POOL.

    Uses a fixed seed so benchmarks are reproducible across runs.

    Args:
        n:    Number of documents to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n random text strings.
    """
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        length = rng.randint(10, 30)
        words = rng.choices(WORD_POOL, k=length)
        texts.append(" ".join(words))
    return texts


def generate_query_texts(n: int, seed: int = 99) -> List[str]:
    """
    Generate n short query strings (3-8 words each).

    Uses a different seed than generate_random_texts so queries
    don't exactly match inserted documents.

    Args:
        n:    Number of queries to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of n random query strings.
    """
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        length = rng.randint(3, 8)
        words = rng.choices(WORD_POOL, k=length)
        queries.append(" ".join(words))
    return queries


def generate_metadata(n: int) -> List[Dict[str, Any]]:
    """
    Generate metadata dicts for n documents.

    Distributes documents evenly across 5 categories
    to simulate a realistic dataset.
    """
    categories = ["technology", "science", "sports", "health", "business"]
    subcategories = {
        "technology": ["ai", "programming", "gadgets", "software"],
        "science": ["physics", "biology", "chemistry", "astronomy"],
        "sports": ["football", "basketball", "cricket", "tennis"],
        "health": ["nutrition", "fitness", "mental_health", "medicine"],
        "business": ["startups", "finance", "marketing", "economy"],
    }
    metadata_list = []
    for i in range(n):
        cat = categories[i % len(categories)]
        sub = subcategories[cat][i % len(subcategories[cat])]
        metadata_list.append({
            "category": cat,
            "subcategory": sub,
            "source": "benchmark",
        })
    return metadata_list


# ═══════════════════════════════════════════════════════════════
# TABLE FORMATTING
# ═══════════════════════════════════════════════════════════════

def format_table(title: str, headers: List[str], rows: List[List[str]]) -> str:
    """
    Format data as a clean ASCII table.

    Args:
        title:   Table title printed above the table.
        headers: Column header strings.
        rows:    List of rows, each a list of string values.

    Returns:
        Multi-line string of the formatted table.
    """
    # Calculate column widths: max of header and all row values
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    # Add padding
    col_widths = [w + 2 for w in col_widths]

    # Build table
    lines = []
    lines.append(f"\n  {title}")
    lines.append("  " + "-" * (sum(col_widths) + len(col_widths) + 1))

    # Header row
    header_str = "  |"
    for h, w in zip(headers, col_widths):
        header_str += f" {h:<{w}}|"
    lines.append(header_str)

    # Separator
    sep_str = "  |"
    for w in col_widths:
        sep_str += "-" * (w + 1) + "|"
    lines.append(sep_str)

    # Data rows
    for row in rows:
        row_str = "  |"
        for val, w in zip(row, col_widths):
            row_str += f" {val:<{w}}|"
        lines.append(row_str)

    lines.append("  " + "-" * (sum(col_widths) + len(col_widths) + 1))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# STORE FACTORY
# ═══════════════════════════════════════════════════════════════

def _make_store(storage_dir: str, use_simple: bool = True) -> VectorStore:
    """
    Create a VectorStore for benchmarking.

    Args:
        storage_dir: Path to the storage directory.
        use_simple:  If True, replace the engine with SimpleEmbeddingEngine
                     for fast BoW-based benchmarks. If False, keep the
                     default engine (real sentence-transformers model).

    Returns:
        Configured VectorStore instance.
    """
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=DIMENSION,
    )
    if use_simple:
        store.embedding_engine = SimpleEmbeddingEngine(dimension=DIMENSION)
    return store


# ═══════════════════════════════════════════════════════════════
# CHECK IF REAL MODEL IS AVAILABLE
# ═══════════════════════════════════════════════════════════════

def is_real_model_available() -> bool:
    """Check if sentence-transformers is installed."""
    engine = EmbeddingEngine()
    return engine._check_availability()


def prewarm_real_model() -> float:
    """
    Load the real sentence-transformers model and return load time.

    This triggers the lazy model load so subsequent benchmarks
    don't include cold-start overhead.

    Returns:
        Model load time in seconds.
    """
    temp_dir = tempfile.mkdtemp(prefix="minivecdb_prewarm_")
    try:
        store = _make_store(temp_dir, use_simple=False)
        start = time.perf_counter()
        store.embedding_engine.encode("warmup sentence for model loading")
        elapsed = time.perf_counter() - start
        store.close()
        return elapsed
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 1: INSERTION THROUGHPUT
# ═══════════════════════════════════════════════════════════════

def bench_insertion(sizes: List[int], use_simple: bool = True) -> List[Dict[str, Any]]:
    """
    Benchmark insertion throughput at different database sizes.

    For each size N, creates a fresh store, generates N random texts,
    and times insert_batch(). Repeated NUM_RUNS times and averaged.

    Args:
        sizes:      List of database sizes to test (e.g., [100, 500, 1000, 2000]).
        use_simple: Whether to use SimpleEmbeddingEngine.

    Returns:
        List of result dicts, one per size.
    """
    results = []

    for size in sizes:
        run_times = []

        for run_idx in range(NUM_RUNS):
            # Use different seed per run so we don't benchmark the same texts
            texts = generate_random_texts(size, seed=42 + run_idx)
            metadata_list = generate_metadata(size)

            temp_dir = tempfile.mkdtemp(prefix=f"minivecdb_bench_insert_{size}_")
            try:
                store = _make_store(temp_dir, use_simple=use_simple)

                start = time.perf_counter()
                store.insert_batch(texts=texts, metadata_list=metadata_list)
                elapsed = time.perf_counter() - start

                run_times.append(elapsed)
                store.close()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        avg_time = sum(run_times) / len(run_times)
        docs_per_sec = size / avg_time if avg_time > 0 else 0

        results.append({
            "size": size,
            "avg_time_s": round(avg_time, 4),
            "docs_per_sec": round(docs_per_sec, 1),
            "run_times_s": [round(t, 4) for t in run_times],
        })

    return results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 2: QUERY LATENCY
# ═══════════════════════════════════════════════════════════════

def bench_query_latency(
    sizes: List[int], use_simple: bool = True
) -> List[Dict[str, Any]]:
    """
    Benchmark search query latency at different database sizes.

    For each size N:
      1. Insert N documents into a fresh store.
      2. Run QUERY_COUNT individual search queries, timing each one.
      3. Repeat NUM_RUNS times.
      4. Aggregate all latencies (NUM_RUNS × QUERY_COUNT) and compute
         average, p50, and p95 percentiles.

    Args:
        sizes:      List of database sizes to test.
        use_simple: Whether to use SimpleEmbeddingEngine.

    Returns:
        List of result dicts, one per size.
    """
    results = []
    queries = generate_query_texts(QUERY_COUNT)

    for size in sizes:
        all_latencies_ms = []

        for run_idx in range(NUM_RUNS):
            texts = generate_random_texts(size, seed=42 + run_idx)
            metadata_list = generate_metadata(size)

            temp_dir = tempfile.mkdtemp(prefix=f"minivecdb_bench_query_{size}_")
            try:
                store = _make_store(temp_dir, use_simple=use_simple)
                store.insert_batch(texts=texts, metadata_list=metadata_list)

                # Time each query individually
                for query in queries:
                    start = time.perf_counter()
                    store.search(query=query, top_k=5, metric="cosine")
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    all_latencies_ms.append(elapsed_ms)

                store.close()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        latency_array = np.array(all_latencies_ms)
        total_queries = len(all_latencies_ms)

        results.append({
            "size": size,
            "total_queries": total_queries,
            "avg_ms": round(float(np.mean(latency_array)), 4),
            "p50_ms": round(float(np.percentile(latency_array, 50)), 4),
            "p95_ms": round(float(np.percentile(latency_array, 95)), 4),
            "min_ms": round(float(np.min(latency_array)), 4),
            "max_ms": round(float(np.max(latency_array)), 4),
        })

    return results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 3: MEMORY USAGE
# ═══════════════════════════════════════════════════════════════

def bench_memory(sizes: List[int]) -> List[Dict[str, Any]]:
    """
    Measure memory usage of the vector matrix at different sizes.

    Memory is engine-independent: both engines produce (N, 384)
    float32 arrays, so we only run this once with SimpleEmbeddingEngine.

    Args:
        sizes: List of database sizes to test.

    Returns:
        List of result dicts, one per size.
    """
    results = []

    for size in sizes:
        texts = generate_random_texts(size)
        metadata_list = generate_metadata(size)

        temp_dir = tempfile.mkdtemp(prefix=f"minivecdb_bench_mem_{size}_")
        try:
            store = _make_store(temp_dir, use_simple=True)
            store.insert_batch(texts=texts, metadata_list=metadata_list)

            nbytes = store._vectors.nbytes
            shape = store._vectors.shape

            results.append({
                "size": size,
                "shape": list(shape),
                "bytes": nbytes,
                "kb": round(nbytes / 1024, 2),
                "mb": round(nbytes / (1024 * 1024), 4),
            })

            store.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


# ═══════════════════════════════════════════════════════════════
# BENCHMARK 4: METRIC COMPARISON
# ═══════════════════════════════════════════════════════════════

def bench_metric_comparison(use_simple: bool = True) -> Dict[str, Any]:
    """
    Compare query latency across all 3 distance metrics at a fixed size.

    Uses N=METRIC_COMPARISON_SIZE (1000). For each metric, runs
    QUERY_COUNT queries × NUM_RUNS times and computes avg/p50/p95.

    Args:
        use_simple: Whether to use SimpleEmbeddingEngine.

    Returns:
        Dict with per-metric results.
    """
    size = METRIC_COMPARISON_SIZE
    queries = generate_query_texts(QUERY_COUNT)
    metric_results = {}

    for metric_name in METRICS:
        all_latencies_ms = []

        for run_idx in range(NUM_RUNS):
            texts = generate_random_texts(size, seed=42 + run_idx)
            metadata_list = generate_metadata(size)

            temp_dir = tempfile.mkdtemp(prefix=f"minivecdb_bench_metric_{metric_name}_")
            try:
                store = _make_store(temp_dir, use_simple=use_simple)
                store.insert_batch(texts=texts, metadata_list=metadata_list)

                for query in queries:
                    start = time.perf_counter()
                    store.search(query=query, top_k=5, metric=metric_name)
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    all_latencies_ms.append(elapsed_ms)

                store.close()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        latency_array = np.array(all_latencies_ms)
        metric_results[metric_name] = {
            "total_queries": len(all_latencies_ms),
            "avg_ms": round(float(np.mean(latency_array)), 4),
            "p50_ms": round(float(np.percentile(latency_array, 50)), 4),
            "p95_ms": round(float(np.percentile(latency_array, 95)), 4),
        }

    return {
        "size": size,
        "metrics": metric_results,
    }


# ═══════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def run_all_benchmarks() -> Dict[str, Any]:
    """
    Run the complete benchmark suite for both embedding engines.

    Execution order:
      1. SimpleBoW insertion throughput
      2. SimpleBoW query latency
      3. Memory usage (engine-independent, run once)
      4. SimpleBoW metric comparison
      5. (If available) Real model pre-warm + insertion + query + metrics

    Returns:
        Full results dict ready for JSON serialisation.
    """
    results: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sizes": SIZES,
            "query_count": QUERY_COUNT,
            "num_runs": NUM_RUNS,
            "dimension": DIMENSION,
            "metrics": METRICS,
        },
    }

    # ----------------------------------------------------------
    # SimpleBoW Engine benchmarks
    # ----------------------------------------------------------
    print_banner("BENCHMARKING: SimpleBoW Engine (bag-of-words fallback)")

    print("\n  [1/4] Insertion throughput...")
    simple_insertion = bench_insertion(SIZES, use_simple=True)
    table = format_table(
        "INSERTION THROUGHPUT — SimpleBoW Engine",
        ["N", "Avg Time (s)", "Docs/sec"],
        [[str(r["size"]), f'{r["avg_time_s"]:.4f}', f'{r["docs_per_sec"]:.1f}']
         for r in simple_insertion],
    )
    print(table)

    print("\n  [2/4] Query latency...")
    simple_query = bench_query_latency(SIZES, use_simple=True)
    table = format_table(
        f"QUERY LATENCY — SimpleBoW Engine ({QUERY_COUNT} queries x {NUM_RUNS} runs)",
        ["N", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Min (ms)", "Max (ms)"],
        [[str(r["size"]), f'{r["avg_ms"]:.4f}', f'{r["p50_ms"]:.4f}',
          f'{r["p95_ms"]:.4f}', f'{r["min_ms"]:.4f}', f'{r["max_ms"]:.4f}']
         for r in simple_query],
    )
    print(table)

    print("\n  [3/4] Memory usage...")
    memory = bench_memory(SIZES)
    table = format_table(
        "MEMORY USAGE — _vectors.nbytes (both engines identical)",
        ["N", "Shape", "Bytes", "KB", "MB"],
        [[str(r["size"]), str(r["shape"]), str(r["bytes"]),
          f'{r["kb"]:.2f}', f'{r["mb"]:.4f}']
         for r in memory],
    )
    print(table)

    print("\n  [4/4] Metric comparison at N=1000...")
    simple_metrics = bench_metric_comparison(use_simple=True)
    table = format_table(
        f"METRIC COMPARISON at N={METRIC_COMPARISON_SIZE} — SimpleBoW Engine",
        ["Metric", "Avg (ms)", "P50 (ms)", "P95 (ms)"],
        [[name, f'{data["avg_ms"]:.4f}', f'{data["p50_ms"]:.4f}', f'{data["p95_ms"]:.4f}']
         for name, data in simple_metrics["metrics"].items()],
    )
    print(table)

    results["simple_engine"] = {
        "name": "SimpleBoW (bag-of-words fallback)",
        "insertion": simple_insertion,
        "query_latency": simple_query,
        "metric_comparison": simple_metrics,
    }
    results["memory_usage"] = memory

    # ----------------------------------------------------------
    # Real SentenceTransformer Engine benchmarks
    # ----------------------------------------------------------
    real_model_ok = is_real_model_available()

    if not real_model_ok:
        print("\n" + "!" * 65)
        print("  WARNING: sentence-transformers not installed.")
        print("  Skipping real model benchmarks.")
        print("  Install with: pip install sentence-transformers")
        print("!" * 65)
        results["real_engine"] = {
            "name": "SentenceTransformer (not available)",
            "skipped": True,
        }
    else:
        print_banner("BENCHMARKING: SentenceTransformer Engine (all-MiniLM-L6-v2)")

        # Pre-warm: load the model once and time it
        print("\n  Pre-warming model (first load triggers download/cache)...")
        model_load_time = prewarm_real_model()
        print(f"  Model loaded in {model_load_time:.2f}s")

        print("\n  [1/3] Insertion throughput...")
        real_insertion = bench_insertion(SIZES, use_simple=False)
        table = format_table(
            "INSERTION THROUGHPUT — SentenceTransformer Engine",
            ["N", "Avg Time (s)", "Docs/sec"],
            [[str(r["size"]), f'{r["avg_time_s"]:.4f}', f'{r["docs_per_sec"]:.1f}']
             for r in real_insertion],
        )
        print(table)

        print("\n  [2/3] Query latency...")
        real_query = bench_query_latency(SIZES, use_simple=False)
        table = format_table(
            f"QUERY LATENCY — SentenceTransformer Engine ({QUERY_COUNT} queries x {NUM_RUNS} runs)",
            ["N", "Avg (ms)", "P50 (ms)", "P95 (ms)", "Min (ms)", "Max (ms)"],
            [[str(r["size"]), f'{r["avg_ms"]:.4f}', f'{r["p50_ms"]:.4f}',
              f'{r["p95_ms"]:.4f}', f'{r["min_ms"]:.4f}', f'{r["max_ms"]:.4f}']
             for r in real_query],
        )
        print(table)

        print("\n  [3/3] Metric comparison at N=1000...")
        real_metrics = bench_metric_comparison(use_simple=False)
        table = format_table(
            f"METRIC COMPARISON at N={METRIC_COMPARISON_SIZE} — SentenceTransformer Engine",
            ["Metric", "Avg (ms)", "P50 (ms)", "P95 (ms)"],
            [[name, f'{data["avg_ms"]:.4f}', f'{data["p50_ms"]:.4f}', f'{data["p95_ms"]:.4f}']
             for name, data in real_metrics["metrics"].items()],
        )
        print(table)

        results["real_engine"] = {
            "name": "SentenceTransformer (all-MiniLM-L6-v2)",
            "model_load_time_s": round(model_load_time, 4),
            "insertion": real_insertion,
            "query_latency": real_query,
            "metric_comparison": real_metrics,
        }

    # ----------------------------------------------------------
    # Side-by-side summary
    # ----------------------------------------------------------
    if real_model_ok:
        print_banner("SIDE-BY-SIDE COMPARISON")
        print_comparison_tables(results)

    return results


def print_comparison_tables(results: Dict[str, Any]) -> None:
    """
    Print side-by-side comparison of both engines.

    Shows insertion throughput and query latency for both engines
    in a single table so differences are immediately visible.
    """
    simple_ins = results["simple_engine"]["insertion"]
    real_ins = results["real_engine"]["insertion"]

    table = format_table(
        "INSERTION THROUGHPUT — Side by Side",
        ["N", "SimpleBoW (doc/s)", "Transformer (doc/s)", "Speedup"],
        [[str(s["size"]),
          f'{s["docs_per_sec"]:.1f}',
          f'{r["docs_per_sec"]:.1f}',
          f'{s["docs_per_sec"] / r["docs_per_sec"]:.1f}x' if r["docs_per_sec"] > 0 else "N/A"]
         for s, r in zip(simple_ins, real_ins)],
    )
    print(table)

    simple_ql = results["simple_engine"]["query_latency"]
    real_ql = results["real_engine"]["query_latency"]

    table = format_table(
        "QUERY LATENCY (avg ms) — Side by Side",
        ["N", "SimpleBoW (ms)", "Transformer (ms)", "Ratio"],
        [[str(s["size"]),
          f'{s["avg_ms"]:.4f}',
          f'{r["avg_ms"]:.4f}',
          f'{r["avg_ms"] / s["avg_ms"]:.1f}x' if s["avg_ms"] > 0 else "N/A"]
         for s, r in zip(simple_ql, real_ql)],
    )
    print(table)


# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════

def save_results(results: Dict[str, Any], path: str = RESULTS_PATH) -> str:
    """
    Save benchmark results as a JSON file.

    Args:
        results: The full results dict from run_all_benchmarks().
        path:    File path to write JSON output.

    Returns:
        The path where results were saved.
    """
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


# ═══════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════

def print_banner(text: str) -> None:
    """Print a prominent section banner."""
    width = 65
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point for the benchmark suite."""
    print("\n" + "+" * 65)
    print("+  MiniVecDB — Performance Benchmark Suite")
    print(f"+  Sizes: {SIZES}  |  Queries: {QUERY_COUNT}  |  Runs: {NUM_RUNS}")
    print(f"+  Engines: SimpleBoW + SentenceTransformer")
    print("+" * 65)

    total_start = time.perf_counter()

    results = run_all_benchmarks()

    # Save JSON results
    json_path = save_results(results)

    total_elapsed = time.perf_counter() - total_start

    print_banner("BENCHMARK COMPLETE")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  Results saved to: {json_path}")
    print()


if __name__ == "__main__":
    main()
