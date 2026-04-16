# MiniVecDB — File: `core/file_processor.py` (File Upload Pipeline)

> **Location**: `minivecdb/core/file_processor.py`
> **Lines**: ~850 | **Dependencies**: `csv`, `pandas`, `openpyxl`
> **Purpose**: Parse uploaded TXT/CSV/Excel files into `(texts, metadata_list)` for `VectorStore.insert_batch()`

---

## Why This File Exists

Users need a shared bulk-ingestion pipeline that works in both interfaces:
- Web UI upload flow (`POST /upload`)
- CLI import flow (`import-file`)

This module is framework-independent (no Flask/argparse coupling) and handles:
1. File validation
2. Text extraction/normalization
3. Chunking policy by file type
4. Metadata expansion per chunk

---

## Pipeline Overview

```
validate_file()
  ├─ file exists
  ├─ size <= 10 MB
  └─ extension in {.txt, .csv, .xlsx, .xls}

extract_from_*
  ├─ TXT:
  │    read text (UTF-8 fallback Latin-1)
  ├─ CSV:
  │    read raw rows with stdlib csv + delimiter sniff
  │    detect header row (or use override)
  │    normalize columns/rows
  │    serialize each data row -> "Column: value | Column2: value2"
  └─ Excel:
       read sheet with pandas header=None
       detect header row (or use override)
       normalize columns/rows
       serialize each data row similarly

chunking
  ├─ TXT: semantic chunking with overlap (chunk_text)
  └─ CSV/Excel: row-first chunking, overlap = 0
       (only split when one serialized row exceeds max_chars)

return (texts, metadata_list)
```

---

## Core Behavior

### Header detection and tabular normalization

For CSV/Excel, raw tables are normalized by:
- scoring likely header rows (`_detect_header_row`) using non-empty density, keyword hits, text-vs-numeric shape, and row continuity
- promoting the selected row to column names
- generating safe unique fallback names (`column_1`, `column_2`, ...)
- dropping fully empty rows/columns
- dropping duplicated first data row if it repeats header labels

Supported manual overrides from callers:
- `header_row`
- `skip_rows`
- `sheet_name` (Excel)

### Row serialization

Each tabular row is serialized deterministically as:

`ColumnA: valueA | ColumnB: valueB | ...`

Only non-empty values are included.  
The same non-empty per-column values are copied into metadata keys.

### Format-specific chunking

| File Type | Chunk Strategy | Overlap |
|----------|----------------|---------|
| TXT | `chunk_text()` sentence/word-aware splitter | configurable (`overlap`) |
| CSV/Excel | row-first chunking (`_chunk_tabular_text`) | forced `0` |

For long tabular rows, splitting happens by field boundary first (`" | "`), then word/hard-cut fallback, still with zero overlap.

---

## Key Functions

### `process_file(file_path, original_filename, max_chars=500, overlap=50, *, header_row=None, skip_rows=None, sheet_name=0)`

Main orchestrator:
- validates file
- calls format-specific extractor
- applies format-specific chunk policy
- expands per-chunk metadata with:
  - `source`
  - `file_type`
  - `row` (tabular)
  - `sheet` (Excel)
  - `chunk_index`
  - `total_chunks`
  - per-column values (tabular)

### `extract_from_csv(...)`

- Reads raw CSV using stdlib `csv` (handles ragged rows by padding)
- Applies optional `skip_rows`
- Detects/uses header row
- Serializes each data row into deterministic text + metadata

### `extract_from_excel(...)`

- Reads sheet with `pandas.read_excel(..., header=None)`
- Applies optional `skip_rows` and `sheet_name`
- Detects/uses header row
- Serializes each data row into deterministic text + metadata

### `chunk_text(...)`

Semantic splitter for TXT with overlap support.

### `_chunk_tabular_text(...)`

Row-preserving tabular splitter with zero overlap.

---

## Validation & Edge Cases

| Scenario | Behavior |
|----------|----------|
| File > 10 MB | `ValueError` with size details |
| Unsupported extension | `ValueError` listing allowed types |
| Empty/whitespace TXT | `ValueError` |
| CSV/Excel with only preamble and no data rows | `ValueError` after normalization |
| Delayed header row (e.g., title + blank + header) | detected automatically |
| Duplicate header copied into first data row | dropped |
| Row with all empty cells | dropped |
| Long tabular row | split with overlap `0` |
| Non-uniform CSV row width | padded safely before DataFrame normalization |

---

## Integration Points

### Web UI (`web/app.py`)

```python
texts, metadata_list = process_file(
    tmp_path,
    original_filename,
    max_chars=500,
    overlap=50,  # used by TXT; tabular remains zero-overlap internally
)
ids = store.insert_batch(texts=texts, metadata_list=metadata_list, collection=collection)
```

### CLI (`cli/main.py`)

```python
texts, metadata_list = process_file(
    file_path,
    original_filename,
    max_chars=args.chunk_size,
    overlap=args.chunk_overlap,
    header_row=args.header_row,
    skip_rows=parsed_skip_rows,
    sheet_name=parsed_sheet,
)
```

---

## Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `MAX_FILE_SIZE_BYTES` | 10,485,760 (10 MB) | Upload size limit |
| `ALLOWED_EXTENSIONS` | `.txt`, `.csv`, `.xlsx`, `.xls` | Supported file types |
| `_HEADER_KEYWORDS` | token list | Header-row scoring hints for tabular files |
