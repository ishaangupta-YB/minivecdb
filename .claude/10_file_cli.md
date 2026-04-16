# MiniVecDB — File: cli/main.py (Command-Line Interface)

> **Location**: `minivecdb/cli/main.py`
> **Lines**: ~805 | **Size**: ~26 KB
> **Purpose**: Terminal interface to MiniVecDB using Python's built-in argparse library

---

## Why This File Exists

The CLI gives users a way to interact with MiniVecDB from the terminal, similar to how `git` has subcommands (`git commit`, `git push`). It provides 10 subcommands for all database operations.

This file is a **thin wrapper** — it parses arguments, calls `VectorStore` methods, and formats output. No business logic lives here.

---

## Architecture

```
Terminal → argparse → build_parser() → parse_args()
                                          │
                                          ▼
                              args.handler(args, store)
                                          │
                         ┌────────────────┼──────────────────┐
                         ▼                ▼                  ▼
                    cmd_insert()    cmd_search()         cmd_stats()
                         │                │                  │
                         ▼                ▼                  ▼
                   store.insert()  store.search()      store.stats()
```

---

## Available Commands

| Command | Purpose | Key Args |
|---------|---------|----------|
| `insert` | Add a document | `--text`, `--metadata`, `--collection` |
| `search` | Semantic search | `--query`, `--top-k`, `--metric`, `--filter`, `--collection` |
| `get` | Retrieve by ID | `--id` |
| `delete` | Remove by ID | `--id` |
| `update` | Modify text/metadata | `--id`, `--text`, `--metadata` |
| `list` | List record IDs | `--collection`, `--limit` |
| `stats` | Database statistics | (none) |
| `collections` | List all collections | (none) |
| `create-collection` | Create new collection | `--name`, `--description` |
| `import-file` | Import & chunk a file (TXT/CSV/Excel) | `--file`, `--collection`, `--chunk-size`, `--chunk-overlap`, `--header-row`, `--skip-rows`, `--sheet` |

### Global Options (apply to all commands)

| Option | Purpose |
|--------|---------|
| `--db-path` | Explicit storage directory (overrides managed runs) |
| `--new-run` | Force a fresh run directory under db_run/ |
| `--run-prefix` | Prefix for run dir names (default: "demo") |
| `--model-cache-path` | Override embedding model cache location |

---

## Usage Examples

```bash
# Insert a document
python cli/main.py insert --text "Python is great for data science"

# Insert with metadata
python cli/main.py insert --text "Einstein's theory" --metadata '{"topic":"physics","year":"1905"}'

# Semantic search
python cli/main.py search --query "machine learning" --top-k 3 --metric cosine

# Search with metadata filter
python cli/main.py search --query "AI" --filter '{"year":{"$gt":"2020"}}'

# Get a specific record
python cli/main.py get --id vec_a1b2c3d4

# Delete a record
python cli/main.py delete --id vec_a1b2c3d4

# Update text (re-embeds the vector)
python cli/main.py update --id vec_a1b2c3d4 --text "Updated text here"

# List record IDs
python cli/main.py list --collection science --limit 10

# Show database stats
python cli/main.py stats

# List collections
python cli/main.py collections

# Create a collection
python cli/main.py create-collection --name papers --description "Research papers"

# Bulk import from file (TXT, CSV, or Excel — auto-detects format)
python cli/main.py import-file --file data/documents.txt --collection papers
python cli/main.py import-file --file data/articles.csv --collection papers --chunk-size 500 --chunk-overlap 50
python cli/main.py import-file --file data/roster.csv --header-row 2 --skip-rows 0,1
python cli/main.py import-file --file data/roster.xlsx --sheet Students --header-row 1

# Force a new run
python cli/main.py --new-run insert --text "Fresh start"
```

---

## Functions — Detailed Breakdown

### Output Helpers

#### `print_table(headers, rows)`
**Lines 60–94** | Custom table formatter for terminal output.

**Algorithm**:
1. Convert all values to strings
2. Compute max width for each column (scanning headers + all rows)
3. Build a format string with left-aligned fields: `{:<10}  {:<20}`
4. Print header, separator line, and data rows

**Why custom?** Avoids importing `tabulate` — keeps dependencies minimal.

#### `print_record(record)`
**Lines 97–109** | Formats a VectorRecord in key-value style with timestamp formatting and vector preview.

#### `format_bytes(num_bytes) → str`
**Lines 112–128** | Converts bytes to human-readable: `1536 → "1.5 KB"`, `1048576 → "1.0 MB"`.

#### `parse_json_arg(value, arg_name) → dict`
**Lines 131–158** | Parses a JSON string from CLI args with clear error messages on malformed input.

---

### Command Handlers

Each handler follows the pattern:
1. Extract arguments from `args` namespace
2. Call the appropriate `VectorStore` method
3. Format and print the result
4. Exit with code 1 on errors

#### `cmd_insert(args, store)`
**Lines 175–192** | Parses optional metadata JSON, calls `store.insert()`, prints the assigned ID.

#### `cmd_search(args, store)`
**Lines 195–239** | Parses filter JSON, calls `store.search()`, formats results as a table with rank, score, ID, collection, and truncated text (50 chars).

#### `cmd_get(args, store)`
**Lines 242–255** | Retrieves and displays a record by ID.

#### `cmd_delete(args, store)`
**Lines 258–270** | Deletes a record and confirms.

#### `cmd_update(args, store)`
**Lines 273–300** | Requires at least one of `--text` or `--metadata`. Calls `store.update()`.

#### `cmd_list(args, store)`
**Lines 303–324** | Lists record IDs with numbered output.

#### `cmd_stats(args, store)`
**Lines 327–344** | Displays database statistics in a formatted panel.

#### `cmd_collections(args, store)`
**Lines 347–377** | Lists all collections in a table with creation dates.

#### `cmd_create_collection(args, store)`
**Lines 380–392** | Creates a new collection.

#### `cmd_import_file(args, store)`
**Lines ~430–530** | Supports TXT, CSV, and Excel files. Delegates to `core.file_processor.process_file()` which validates the file (size ≤10 MB, supported extension), performs robust tabular header normalization for CSV/Excel, serializes each tabular row as deterministic `Column: value` text, and chunks with format-specific policy:
- TXT: semantic chunking with `--chunk-overlap`
- CSV/Excel: row-first chunking with zero overlap

Additional tabular overrides:
- `--header-row` (force header row index)
- `--skip-rows` (single int or comma list)
- `--sheet` (Excel sheet index/name)

---

### Parser Setup

#### `build_parser() → ArgumentParser`
**Lines 454–663** | Constructs the full argparse parser with all subcommands.

Uses `subparsers.add_parser()` for each command, with:
- `add_argument()` for command-specific flags
- `set_defaults(handler=cmd_xxx)` to link each subparser to its handler function

---

### Main Entry Point

#### `main()`
**Lines 670–718**

```python
def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    with VectorStore(
        storage_path=args.db_path,
        new_run=args.new_run,
        run_prefix=args.run_prefix,
        model_cache_path=args.model_cache_path,
    ) as store:
        args.handler(args, store)
```

**Key design decisions**:
- `with VectorStore(...) as store:` ensures `close()` is always called
- `args.handler(args, store)` dynamically dispatches to the right command function
- `except KeyboardInterrupt` → clean exit with code 130
- `except Exception` → prints a user-friendly error (not a Python traceback)
