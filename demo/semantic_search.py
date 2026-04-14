"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Semantic Search Demo                             ║
║  File: minivecdb/demo/semantic_search.py                      ║
║                                                               ║
║  A standalone end-to-end demo that:                           ║
║    1. Loads the curated sample dataset (150+ documents)       ║
║    2. Creates a VectorStore and bulk-inserts all documents     ║
║    3. Runs 10 diverse search queries and prints results        ║
║    4. Demonstrates metadata-filtered search (science only)    ║
║    5. Prints database statistics                               ║
║    6. Shows that similar-meaning queries return similar results║
║                                                               ║
║  Run:  python -m minivecdb.demo.semantic_search               ║
╚═══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time

# ---------------------------------------------------------------
# Ensure the project root is on sys.path so imports work
# whether you run this as a module or directly.
# ---------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.vector_store import VectorStore
from data.sample_dataset import load_dataset


# ═══════════════════════════════════════════════════════════════
# HELPER: Pretty-print search results
# ═══════════════════════════════════════════════════════════════

def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 65
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_results(results: list, show_metadata: bool = False) -> None:
    """
    Print search results in a readable format.

    Args:
        results:       List of SearchResult objects from VectorStore.search().
        show_metadata: If True, also print each result's metadata dict.
    """
    if not results:
        print("  (no results found)")
        return

    for r in results:
        # Truncate long text to keep output clean
        text_preview = r.record.text
        if len(text_preview) > 120:
            text_preview = text_preview[:117] + "..."

        print(f"  #{r.rank}  [score: {r.score:.4f}]  {text_preview}")
        if show_metadata:
            print(f"       metadata: {r.record.metadata}")


# ═══════════════════════════════════════════════════════════════
# STEP 1: Load dataset
# ═══════════════════════════════════════════════════════════════

def load_and_summarize_dataset() -> list:
    """Load the dataset and print a summary of what we got."""
    print_header("STEP 1: Loading Dataset")

    dataset = load_dataset()
    print(f"  Loaded {len(dataset)} documents")

    # Count documents per category
    categories: dict = {}
    for doc in dataset:
        cat = doc["metadata"]["category"]
        sub = doc["metadata"]["subcategory"]
        categories.setdefault(cat, {"count": 0, "subcategories": set()})
        categories[cat]["count"] += 1
        categories[cat]["subcategories"].add(sub)

    print("\n  Category breakdown:")
    for cat in sorted(categories):
        info = categories[cat]
        subs = ", ".join(sorted(info["subcategories"]))
        print(f"    {cat:12s} : {info['count']:3d} docs  ({subs})")

    return dataset


# ═══════════════════════════════════════════════════════════════
# STEP 2: Create VectorStore and bulk-insert
# ═══════════════════════════════════════════════════════════════

def create_store_and_insert(dataset: list) -> VectorStore:
    """
    Create a fresh VectorStore and insert all documents using batch insert.

    Batch insert is MUCH faster than individual inserts because:
      - The embedding model processes all texts in one forward pass.
      - We only write vectors.npy and id_mapping.json once at the end.

    Args:
        dataset: List of dicts with "text" and "metadata" keys.

    Returns:
        The populated VectorStore instance.
    """
    print_header("STEP 2: Creating VectorStore & Bulk Inserting")

    # Create a new run so each demo gets a clean database
    store = VectorStore(new_run=True, run_prefix="demo_search")
    print(f"  Storage path: {store.storage_path}")
    print(f"  Embedding model: {getattr(store.embedding_engine, 'model_name', 'unknown')}")

    # Separate texts and metadata for batch insert
    texts = [doc["text"] for doc in dataset]
    metadata_list = [doc["metadata"] for doc in dataset]

    print(f"\n  Inserting {len(texts)} documents (batch mode)...")
    start_time = time.time()

    ids = store.insert_batch(texts=texts, metadata_list=metadata_list)

    elapsed = time.time() - start_time
    print(f"  Inserted {len(ids)} documents in {elapsed:.2f}s")
    print(f"  Average: {elapsed / len(ids) * 1000:.1f}ms per document")

    return store


# ═══════════════════════════════════════════════════════════════
# STEP 3: Run 10 example search queries
# ═══════════════════════════════════════════════════════════════

def run_example_queries(store: VectorStore) -> None:
    """
    Run 10 diverse queries across all 5 categories and print results.

    These queries test that the semantic search finds relevant documents
    even when the query uses different words than the stored text.
    For example, "How does artificial intelligence work?" should match
    documents about AI, machine learning, neural networks, etc.
    """
    print_header("STEP 3: Running 10 Example Search Queries")

    queries = [
        # Technology queries
        "How does artificial intelligence work?",
        "What are the best programming languages to learn?",
        # Science queries
        "What is quantum computing?",
        "How do black holes form in space?",
        # Sports queries
        "How to improve at basketball?",
        "Who are the greatest cricket players of all time?",
        # Health queries
        "What are the health benefits of exercise?",
        "How to deal with stress and anxiety?",
        # Business queries
        "Latest trends in stock market",
        "How do startups get funding?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n  Query {i}/{len(queries)}: \"{query}\"")
        print("  " + "-" * 55)

        results = store.search(query=query, top_k=3, metric="cosine")
        print_results(results)


# ═══════════════════════════════════════════════════════════════
# STEP 4: Metadata-filtered search (science category only)
# ═══════════════════════════════════════════════════════════════

def run_filtered_search(store: VectorStore) -> None:
    """
    Demonstrate metadata pre-filtering.

    MiniVecDB supports filtering by metadata BEFORE computing similarity.
    This means:
      1. SQL query finds record IDs where category = "science"
      2. Only those vectors are compared against the query
      3. Results are guaranteed to be from the science category

    This is much more efficient than searching everything and then
    filtering, especially when you have millions of records.
    """
    print_header("STEP 4: Filtered Search (Science Category Only)")

    science_queries = [
        "How do vaccines protect the human body?",
        "What causes climate change?",
        "Explain the theory of relativity",
    ]

    for query in science_queries:
        print(f"\n  Query: \"{query}\"")
        print(f"  Filter: category = Science")
        print("  " + "-" * 55)

        results = store.search(
            query=query,
            top_k=3,
            metric="cosine",
            filters={"category": "Science"},
        )
        print_results(results, show_metadata=True)


# ═══════════════════════════════════════════════════════════════
# STEP 5: Database statistics
# ═══════════════════════════════════════════════════════════════

def print_database_stats(store: VectorStore) -> None:
    """Print a summary of the database state."""
    print_header("STEP 5: Database Statistics")

    db_stats = store.stats()
    memory_kb = db_stats.memory_usage_bytes / 1024

    print(f"  Total records:     {db_stats.total_records}")
    print(f"  Total collections: {db_stats.total_collections}")
    print(f"  Vector dimension:  {db_stats.dimension}")
    print(f"  Memory (vectors):  {memory_kb:.1f} KB ({db_stats.memory_usage_bytes} bytes)")
    print(f"  Embedding model:   {db_stats.embedding_model}")
    print(f"  Storage path:      {db_stats.storage_path}")
    print(f"  SQLite database:   {db_stats.db_file}")


# ═══════════════════════════════════════════════════════════════
# STEP 6: Similar-meaning queries return similar results
# ═══════════════════════════════════════════════════════════════

def demonstrate_semantic_similarity(store: VectorStore) -> None:
    """
    Show that queries with the SAME MEANING but DIFFERENT WORDS
    return the same (or very similar) top results.

    This is the core value of vector search over keyword search:
      - Keyword search: "ML" won't match "machine learning"
      - Vector search:  "ML" and "machine learning" produce
                        nearly identical embedding vectors

    We test 3 pairs of semantically equivalent queries and show
    that their top results overlap significantly.
    """
    print_header("STEP 6: Semantic Similarity Demo")
    print("  Queries with the same meaning should return similar results.")
    print("  This is what makes vector search better than keyword search!\n")

    # Pairs of queries that mean the same thing in different words
    query_pairs = [
        (
            "How does machine learning work?",
            "Explain the fundamentals of AI and neural networks",
        ),
        (
            "Tips for staying healthy and fit",
            "What should I do to improve my physical wellness?",
        ),
        (
            "How to start a new business?",
            "What are the steps for launching a startup company?",
        ),
    ]

    for pair_num, (query_a, query_b) in enumerate(query_pairs, 1):
        print(f"  --- Pair {pair_num} ---")

        # Search with query A
        print(f"  Query A: \"{query_a}\"")
        results_a = store.search(query=query_a, top_k=3, metric="cosine")
        ids_a = {r.record.id for r in results_a}
        print_results(results_a)

        # Search with query B
        print(f"\n  Query B: \"{query_b}\"")
        results_b = store.search(query=query_b, top_k=3, metric="cosine")
        ids_b = {r.record.id for r in results_b}
        print_results(results_b)

        # Calculate overlap
        overlap = ids_a & ids_b
        overlap_pct = len(overlap) / max(len(ids_a), len(ids_b)) * 100
        print(f"\n  Overlap: {len(overlap)}/3 results in common ({overlap_pct:.0f}%)")
        print()


# ═══════════════════════════════════════════════════════════════
# MAIN: Run the full demo end-to-end
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    """
    Run the complete semantic search demo.

    This function orchestrates all 6 steps:
      1. Load and summarize the dataset
      2. Create VectorStore and bulk-insert documents
      3. Run 10 example queries across all categories
      4. Demonstrate metadata-filtered search
      5. Print database statistics
      6. Show semantic similarity (different words, same meaning)
    """
    print("\n" + "+" * 65)
    print("+  MiniVecDB — Semantic Search Demo")
    print("+  A mini vector database built from scratch")
    print("+" * 65)

    total_start = time.time()

    # Step 1: Load dataset
    dataset = load_and_summarize_dataset()

    # Step 2: Create store and insert all documents
    store = create_store_and_insert(dataset)

    try:
        # Step 3: Run 10 example queries
        run_example_queries(store)

        # Step 4: Filtered search (science only)
        run_filtered_search(store)

        # Step 5: Database stats
        print_database_stats(store)

        # Step 6: Semantic similarity demo
        demonstrate_semantic_similarity(store)

    finally:
        # Always close the store to save data and release SQLite connection
        store.close()

    total_elapsed = time.time() - total_start
    print_header("DEMO COMPLETE")
    print(f"  Total time: {total_elapsed:.2f}s")
    print(f"  All data saved to: {store.storage_path}")
    print(f"  You can re-query this data using the CLI or web UI.\n")


if __name__ == "__main__":
    main()
