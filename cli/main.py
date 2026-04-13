"""
+===============================================================+
|  MiniVecDB -- Command-Line Interface (CLI)                     |
|  File: minivecdb/cli/main.py                                   |
|                                                                |
|  This module provides a terminal interface to MiniVecDB using  |
|  Python's built-in argparse library.                           |
|                                                                |
|  Commands:                                                      |
|    insert            Insert a document into the database        |
|    search            Semantic similarity search                 |
|    get               Retrieve a record by ID                    |
|    delete            Delete a record by ID                      |
|    update            Update a record's text and/or metadata     |
|    list              List record IDs in a collection            |
|    stats             Show database statistics                   |
|    collections       List all collections with record counts    |
|    create-collection Create a new collection                    |
|    import-file       Bulk import documents from a text file     |
|                                                                |
|  Usage:                                                         |
|    python cli/main.py insert --text "Hello world"              |
|    python cli/main.py search --query "greeting"                |
|    python cli/main.py stats                                    |
|                                                                |
|  HOW ARGPARSE WORKS:                                           |
|    argparse is Python's built-in library for parsing command-   |
|    line arguments.  We create a "parser" with subcommands      |
|    (like git has "git commit", "git push", etc.).  Each sub-   |
|    command gets its own set of arguments.  argparse handles     |
|    --help automatically, validates types, and gives nice error  |
|    messages when users provide wrong arguments.                 |
+===============================================================+
"""

import argparse
import json
import sys
import os
import time

# ---------------------------------------------------------------
# Add the project root to Python's import path so we can import
# core/, storage/, and ARCHITECTURE.py regardless of where the
# script is run from.
# ---------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import VectorStore


# ===============================================================
# OUTPUT FORMATTING HELPERS
# ===============================================================
# These functions produce aligned, readable terminal output.
# We build our own table formatter rather than pulling in a
# third-party library like tabulate -- keeps dependencies minimal.
# ===============================================================

def print_table(headers: list, rows: list) -> None:
    """
    Print a nicely aligned text table to the terminal.

    Algorithm:
      1. Compute the maximum width for each column by scanning
         both the headers and every row.
      2. Print the header row padded to those widths.
      3. Print a separator line of dashes.
      4. Print each data row padded to the same widths.

    Args:
        headers: List of column header strings.
        rows:    List of lists, each inner list is one table row.
                 Values are converted to strings automatically.
    """
    # Convert all values to strings so we can measure their width.
    str_rows = [[str(v) for v in row] for row in rows]

    # Calculate the widest value in each column (including header).
    col_widths = [len(h) for h in headers]
    for row in str_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    # Build a format string like "{:<10}  {:<20}  {:<8}"
    # The ":<N" means left-aligned, padded to N characters.
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    # Print header, separator, and rows.
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in str_rows:
        print(fmt.format(*row))


def print_record(record) -> None:
    """
    Print a single VectorRecord in a readable key-value format.

    Args:
        record: A VectorRecord dataclass instance.
    """
    print(f"  ID:         {record.id}")
    print(f"  Text:       {record.text}")
    print(f"  Collection: {record.collection}")
    print(f"  Created:    {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.created_at))}")
    print(f"  Metadata:   {json.dumps(record.metadata)}")
    print(f"  Vector:     [{record.vector[0]:.4f}, {record.vector[1]:.4f}, ... ] ({len(record.vector)} dims)")


def format_bytes(num_bytes: int) -> str:
    """
    Convert a byte count into a human-readable string.

    Uses binary units: 1 KB = 1024 bytes.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string like "1.50 MB" or "384 B".
    """
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def parse_json_arg(value: str, arg_name: str) -> dict:
    """
    Parse a JSON string from a CLI argument into a Python dict.

    Gives a clear error message if the JSON is malformed, instead
    of dumping a raw Python traceback.

    Args:
        value:    The raw string from the command line.
        arg_name: The argument name (for error messages).

    Returns:
        Parsed dict.

    Raises:
        SystemExit: If the JSON is invalid (prints error and exits).
    """
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        print(f"Error: --{arg_name} must be valid JSON: {exc}")
        sys.exit(1)

    if not isinstance(parsed, dict):
        print(f"Error: --{arg_name} must be a JSON object (dict), got {type(parsed).__name__}")
        sys.exit(1)

    return parsed


# ===============================================================
# COMMAND HANDLERS
# ===============================================================
# Each function below handles one CLI subcommand.  They all
# receive the parsed argparse.Namespace object (args) and a
# VectorStore instance (store).
#
# Pattern:
#   1. Extract arguments from args
#   2. Call the appropriate VectorStore method
#   3. Format and print the result
#   4. Handle errors gracefully (print message, not traceback)
# ===============================================================

def cmd_insert(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'insert' subcommand.

    Inserts a single document into the database and prints the
    assigned record ID.
    """
    # Parse optional metadata JSON string into a dict.
    metadata = {}
    if args.metadata:
        metadata = parse_json_arg(args.metadata, "metadata")

    record_id = store.insert(
        text=args.text,
        metadata=metadata,
        collection=args.collection,
    )
    print(f"Inserted record: {record_id}")


def cmd_search(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'search' subcommand.

    Runs a semantic similarity search and prints ranked results
    in a table format.
    """
    # Parse optional filter JSON.
    filters = None
    if args.filter:
        filters = parse_json_arg(args.filter, "filter")

    results = store.search(
        query=args.query,
        top_k=args.top_k,
        metric=args.metric,
        filters=filters,
        collection=args.collection,
    )

    if not results:
        print("No results found.")
        return

    # Print a results table with rank, score, ID, and a text preview.
    print(f"\nSearch results for: \"{args.query}\"")
    print(f"Metric: {args.metric} | Top-K: {args.top_k}\n")

    headers = ["Rank", "Score", "ID", "Collection", "Text"]
    rows = []
    for r in results:
        # Truncate text to 50 chars so the table stays readable.
        text_preview = r.record.text[:50]
        if len(r.record.text) > 50:
            text_preview += "..."
        rows.append([
            f"#{r.rank}",
            f"{r.score:.6f}",
            r.record.id,
            r.record.collection,
            text_preview,
        ])

    print_table(headers, rows)
    print(f"\n{len(results)} result(s) returned.")


def cmd_get(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'get' subcommand.

    Retrieves and displays a specific record by its ID.
    """
    record = store.get(args.id)

    if record is None:
        print(f"Error: No record found with ID '{args.id}'")
        sys.exit(1)

    print(f"\nRecord details:")
    print_record(record)


def cmd_delete(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'delete' subcommand.

    Deletes a record by ID and confirms the deletion.
    """
    deleted = store.delete(args.id)

    if deleted:
        print(f"Deleted record: {args.id}")
    else:
        print(f"Error: No record found with ID '{args.id}'")
        sys.exit(1)


def cmd_update(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'update' subcommand.

    Updates an existing record's text and/or metadata.
    At least one of --text or --metadata must be provided.
    """
    # Must provide at least one thing to update.
    if args.text is None and args.metadata is None:
        print("Error: Provide --text and/or --metadata to update.")
        sys.exit(1)

    # Parse optional metadata JSON.
    metadata = None
    if args.metadata is not None:
        metadata = parse_json_arg(args.metadata, "metadata")

    updated = store.update(
        id=args.id,
        text=args.text,
        metadata=metadata,
    )

    if updated:
        print(f"Updated record: {args.id}")
    else:
        print(f"Error: No record found with ID '{args.id}'")
        sys.exit(1)


def cmd_list(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'list' subcommand.

    Lists record IDs in a collection (or all collections).
    """
    ids = store.list_ids(
        collection=args.collection,
        limit=args.limit,
    )

    if not ids:
        collection_msg = f" in collection '{args.collection}'" if args.collection else ""
        print(f"No records found{collection_msg}.")
        return

    collection_msg = f" in '{args.collection}'" if args.collection else ""
    print(f"\nRecord IDs{collection_msg} (showing {len(ids)}):\n")
    for i, record_id in enumerate(ids, 1):
        print(f"  {i:>4}. {record_id}")

    print(f"\n{len(ids)} record(s) listed.")


def cmd_stats(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'stats' subcommand.

    Shows overall database statistics.
    """
    s = store.stats()

    print("\n  MiniVecDB Statistics")
    print("  " + "=" * 40)
    print(f"  Total records:     {s.total_records}")
    print(f"  Total collections: {s.total_collections}")
    print(f"  Vector dimension:  {s.dimension}")
    print(f"  Memory usage:      {format_bytes(s.memory_usage_bytes)}")
    print(f"  Embedding model:   {s.embedding_model}")
    print(f"  Storage path:      {s.storage_path}")
    print(f"  Database file:     {s.db_file}")
    print()


def cmd_collections(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'collections' subcommand.

    Lists all collections with their record counts.
    """
    collections = store.list_collections()

    if not collections:
        print("No collections found.")
        return

    print(f"\nCollections ({len(collections)}):\n")

    headers = ["Name", "Records", "Dimension", "Description", "Created"]
    rows = []
    for col in collections:
        created = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(col.created_at),
        )
        rows.append([
            col.name,
            col.count,
            col.dimension,
            col.description or "(none)",
            created,
        ])

    print_table(headers, rows)
    print()


def cmd_create_collection(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'create-collection' subcommand.

    Creates a new collection with an optional description.
    """
    info = store.create_collection(
        name=args.name,
        description=args.description or "",
    )
    print(f"Created collection: '{info.name}'")
    if info.description:
        print(f"  Description: {info.description}")


def cmd_import_file(args: argparse.Namespace, store: VectorStore) -> None:
    """
    Handle the 'import-file' subcommand.

    Reads a text file (one document per line) and bulk-inserts
    all non-empty lines into the database.

    Uses insert_batch() for speed — all texts are embedded in one
    pass through the neural network, which is 10-50x faster than
    inserting them one by one.
    """
    file_path = args.file

    # --- Validate the file exists ---
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # --- Read lines and filter out blanks ---
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"Error: File is empty or contains only blank lines: {file_path}")
        sys.exit(1)

    print(f"Importing {len(lines)} document(s) from: {file_path}")
    print(f"Collection: {args.collection}")

    # --- Bulk insert using insert_batch() ---
    ids = store.insert_batch(
        texts=lines,
        collection=args.collection,
    )

    print(f"Successfully imported {len(ids)} document(s).")

    # Show a few sample IDs so the user can verify.
    preview_count = min(5, len(ids))
    print(f"\nFirst {preview_count} IDs:")
    for record_id in ids[:preview_count]:
        print(f"  {record_id}")
    if len(ids) > preview_count:
        print(f"  ... and {len(ids) - preview_count} more")


# ===============================================================
# ARGPARSE SETUP
# ===============================================================
# We use argparse "subparsers" to create subcommands.  This is
# the same pattern used by git (git commit, git push, etc.) and
# pip (pip install, pip freeze, etc.).
#
# Structure:
#   1. Create a top-level parser with global options.
#   2. Add a subparser group for commands.
#   3. For each command, add arguments and set a handler function.
# ===============================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the argparse parser with all subcommands.

    Returns:
        Configured ArgumentParser ready to parse sys.argv.
    """
    # --- Top-level parser ---
    # This defines the global options and the program description.
    parser = argparse.ArgumentParser(
        prog="minivecdb",
        description="MiniVecDB -- A mini vector database built from scratch.",
        epilog="Example: python cli/main.py insert --text \"Hello world\"",
    )

    # Global options for storage/cache behavior.
    # Every subcommand inherits these because they are on top-level parser.
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=(
            "Explicit storage directory. If omitted, MiniVecDB uses "
            "managed storage under ./db_run/ and reuses the active run."
        ),
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Force a new unique run directory under ./db_run/ "
            "(ignored when --db-path is provided)."
        ),
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="demo",
        help=(
            "Prefix for managed run directory names "
            "(default: demo -> demo_<timestamp>_<random>)."
        ),
    )
    parser.add_argument(
        "--model-cache-path",
        type=str,
        default=None,
        help=(
            "Optional embedding model cache directory. "
            "Default is ./db_run/model_cache/huggingface."
        ),
    )

    # --- Subcommand group ---
    # dest="command" means the chosen subcommand name is stored in
    # args.command after parsing.
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands (use <command> --help for details)",
    )

    # ----- insert -----
    p_insert = subparsers.add_parser(
        "insert",
        help="Insert a document into the database",
    )
    p_insert.add_argument(
        "--text", required=True,
        help="The document text to insert",
    )
    p_insert.add_argument(
        "--metadata", default=None,
        help='JSON metadata, e.g. \'{"category":"science"}\'',
    )
    p_insert.add_argument(
        "--collection", default="default",
        help="Target collection (default: default)",
    )
    p_insert.set_defaults(handler=cmd_insert)

    # ----- search -----
    p_search = subparsers.add_parser(
        "search",
        help="Search for similar documents",
    )
    p_search.add_argument(
        "--query", required=True,
        help="The search query text",
    )
    p_search.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)",
    )
    p_search.add_argument(
        "--metric", default="cosine",
        choices=["cosine", "euclidean", "dot"],
        help="Similarity metric (default: cosine)",
    )
    p_search.add_argument(
        "--filter", default=None,
        help='JSON metadata filter, e.g. \'{"category":"science"}\'',
    )
    p_search.add_argument(
        "--collection", default=None,
        help="Restrict search to a specific collection",
    )
    p_search.set_defaults(handler=cmd_search)

    # ----- get -----
    p_get = subparsers.add_parser(
        "get",
        help="Retrieve a record by ID",
    )
    p_get.add_argument(
        "--id", required=True,
        help="The record ID to retrieve",
    )
    p_get.set_defaults(handler=cmd_get)

    # ----- delete -----
    p_delete = subparsers.add_parser(
        "delete",
        help="Delete a record by ID",
    )
    p_delete.add_argument(
        "--id", required=True,
        help="The record ID to delete",
    )
    p_delete.set_defaults(handler=cmd_delete)

    # ----- update -----
    p_update = subparsers.add_parser(
        "update",
        help="Update a record's text and/or metadata",
    )
    p_update.add_argument(
        "--id", required=True,
        help="The record ID to update",
    )
    p_update.add_argument(
        "--text", default=None,
        help="New text (re-embeds the vector)",
    )
    p_update.add_argument(
        "--metadata", default=None,
        help='New metadata JSON (replaces all existing metadata)',
    )
    p_update.set_defaults(handler=cmd_update)

    # ----- list -----
    p_list = subparsers.add_parser(
        "list",
        help="List record IDs in a collection",
    )
    p_list.add_argument(
        "--collection", default=None,
        help="Filter by collection (default: all collections)",
    )
    p_list.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of IDs to show (default: 20)",
    )
    p_list.set_defaults(handler=cmd_list)

    # ----- stats -----
    p_stats = subparsers.add_parser(
        "stats",
        help="Show database statistics",
    )
    p_stats.set_defaults(handler=cmd_stats)

    # ----- collections -----
    p_collections = subparsers.add_parser(
        "collections",
        help="List all collections with record counts",
    )
    p_collections.set_defaults(handler=cmd_collections)

    # ----- create-collection -----
    p_create_col = subparsers.add_parser(
        "create-collection",
        help="Create a new collection",
    )
    p_create_col.add_argument(
        "--name", required=True,
        help="Collection name (must be unique)",
    )
    p_create_col.add_argument(
        "--description", default="",
        help='Optional description, e.g. "Research papers"',
    )
    p_create_col.set_defaults(handler=cmd_create_collection)

    # ----- import-file -----
    p_import = subparsers.add_parser(
        "import-file",
        help="Bulk import documents from a text file (one per line)",
    )
    p_import.add_argument(
        "--file", required=True,
        help="Path to the text file to import",
    )
    p_import.add_argument(
        "--collection", default="default",
        help="Target collection (default: default)",
    )
    p_import.set_defaults(handler=cmd_import_file)

    return parser


# ===============================================================
# MAIN ENTRY POINT
# ===============================================================

def main() -> None:
    """
    Parse CLI arguments, open the VectorStore, run the command.

        The flow is:
            1. Parse command-line arguments with argparse.
            2. Open a VectorStore using explicit --db-path or managed db_run.
            3. Dispatch to the appropriate command handler.
            4. Close the VectorStore (always, even on error).
            5. If any error occurs, print a friendly message and exit.
    """
    parser = build_parser()
    args = parser.parse_args()

    # If no subcommand was given, print help and exit.
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # --- Open the VectorStore ---
    # The `with` block ensures close() is always called, even if
    # an exception occurs.  This saves vectors to disk and releases
    # the SQLite connection.
    try:
        if args.db_path is not None and args.new_run:
            print("Note: --new-run ignored because --db-path was provided.")

        with VectorStore(
            storage_path=args.db_path,
            new_run=args.new_run,
            run_prefix=args.run_prefix,
            model_cache_path=args.model_cache_path,
        ) as store:
            # Dispatch to the command handler function.
            # Each subparser has a `handler` default set via set_defaults().
            args.handler(args, store)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        # Print a user-friendly error message instead of a full traceback.
        # This is important for a CLI tool -- users don't want to see
        # Python internals, just what went wrong.
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
