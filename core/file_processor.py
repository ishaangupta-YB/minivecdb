"""
+===============================================================+
|  MiniVecDB -- File Processor (Upload → Extract → Chunk)       |
|  File: minivecdb/core/file_processor.py                       |
|                                                               |
|  Handles file uploads for bulk indexing into the vector store. |
|  Supports three file types:                                   |
|    - TXT  → full-text chunking                                |
|    - CSV  → row-level extraction via pandas                   |
|    - Excel (.xlsx/.xls) → same as CSV via pandas + openpyxl   |
|                                                               |
|  The module is pure-Python with no Flask or CLI dependencies   |
|  so both interfaces can reuse the same pipeline.              |
|                                                               |
|  Pipeline:  validate → extract → chunk → return (texts, meta) |
+===============================================================+
"""

import csv
import io
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS: set = {".txt", ".csv", ".xlsx", ".xls"}

# Tokens used to identify likely header rows in messy CSV/Excel files.
_HEADER_KEYWORDS: List[str] = [
    "name",
    "id",
    "no",
    "number",
    "mail",
    "email",
    "phone",
    "mobile",
    "enrollment",
    "roll",
    "session",
    "date",
    "category",
    "type",
    "title",
    "description",
]

# Sentence-ending patterns used by the chunker to find natural
# split points.  Order matters: we try the most specific first.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


# ===============================================================
# TEXT CHUNKING
# ===============================================================

def chunk_text(
    text: str,
    max_chars: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into chunks of at most *max_chars* characters.

    Strategy (best-effort semantic boundaries):
      1. Split on sentence boundaries (. ! ? followed by whitespace,
         or newlines) to get "sentences."
      2. Accumulate sentences into a buffer.  When adding the next
         sentence would exceed *max_chars*, finalize the current
         chunk and seed a new one with the last *overlap* characters
         of the previous chunk (so neighbouring chunks share context).
      3. If a single sentence still exceeds *max_chars*, fall back to
         word-boundary splitting (spaces).
      4. If a single *word* exceeds *max_chars*, hard-cut it at
         *max_chars* so every chunk is guaranteed <= max_chars.

    Args:
        text:      The full text to chunk.
        max_chars: Maximum characters per chunk (hard ceiling).
        overlap:   Number of trailing characters from the previous
                   chunk to prepend to the next one.

    Returns:
        List of non-empty chunk strings, each <= max_chars.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= max_chars:
        raise ValueError("overlap must be strictly less than max_chars.")

    text = text.strip()
    if not text:
        return []

    # Fast path: entire text fits in one chunk.
    if len(text) <= max_chars:
        return [text]

    # --- Phase 1: split into sentence-level fragments -----------
    sentences = _split_sentences(text)

    # --- Phase 2: accumulate into chunks, respecting max_chars --
    chunks: List[str] = []
    buffer = ""

    for sentence in sentences:
        # If a single sentence is too long, break it further.
        if len(sentence) > max_chars:
            # Flush current buffer first.
            if buffer:
                chunks.append(buffer)
                buffer = _overlap_seed(buffer, overlap)
            for sub in _split_long_segment(sentence, max_chars):
                if len(buffer) + len(sub) <= max_chars:
                    buffer += sub
                else:
                    if buffer:
                        chunks.append(buffer)
                        buffer = _overlap_seed(buffer, overlap)
                    # sub itself might still exceed max_chars after
                    # the overlap seed is prepended.
                    while len(buffer) + len(sub) > max_chars:
                        room = max_chars - len(buffer)
                        if room <= 0:
                            chunks.append(buffer)
                            buffer = _overlap_seed(buffer, overlap)
                            room = max_chars - len(buffer)
                        chunks.append(buffer + sub[:room])
                        leftover = sub[:room]
                        buffer = _overlap_seed(buffer + leftover, overlap)
                        sub = sub[room:]
                    buffer += sub
            continue

        # Normal case: sentence fits inside max_chars on its own.
        candidate = buffer + sentence
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            # Flush and start a new chunk seeded with overlap.
            if buffer:
                chunks.append(buffer)
                buffer = _overlap_seed(buffer, overlap)
            # After seeding, if sentence still doesn't fit we must
            # break it (shouldn't happen unless overlap is very large).
            if len(buffer) + len(sentence) <= max_chars:
                buffer += sentence
            else:
                for sub in _split_long_segment(sentence, max_chars - len(buffer)):
                    if len(buffer) + len(sub) <= max_chars:
                        buffer += sub
                    else:
                        chunks.append(buffer)
                        buffer = _overlap_seed(buffer, overlap) + sub

    # Flush remaining buffer.
    if buffer.strip():
        chunks.append(buffer)

    # Final safety: guarantee every chunk <= max_chars and non-empty.
    result: List[str] = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        if len(c) <= max_chars:
            result.append(c)
        else:
            # Hard-cut fallback (should rarely trigger).
            for i in range(0, len(c), max_chars):
                piece = c[i : i + max_chars].strip()
                if piece:
                    result.append(piece)
    return result


def _split_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries, keeping the delimiters attached."""
    parts = _SENTENCE_BOUNDARY_RE.split(text)
    # Re-attach the whitespace/newline that was consumed by the split
    # so that concatenating all parts reproduces something close to
    # the original (modulo minor whitespace differences).
    sentences: List[str] = []
    for p in parts:
        stripped = p.strip()
        if stripped:
            sentences.append(stripped + " ")
    return sentences


def _split_long_segment(segment: str, max_chars: int) -> List[str]:
    """Break a segment that exceeds *max_chars* on word boundaries,
    falling back to hard-cut when a single word is too long."""
    if max_chars < 1:
        max_chars = 1
    words = segment.split(" ")
    pieces: List[str] = []
    buf = ""
    for word in words:
        # A single word longer than max_chars → hard-cut it.
        if len(word) > max_chars:
            if buf:
                pieces.append(buf)
                buf = ""
            for i in range(0, len(word), max_chars):
                pieces.append(word[i : i + max_chars])
            continue

        candidate = (buf + " " + word) if buf else word
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                pieces.append(buf)
            buf = word
    if buf:
        pieces.append(buf)
    return pieces


def _overlap_seed(prev_chunk: str, overlap: int) -> str:
    """Return the last *overlap* characters of *prev_chunk* to seed
    the next chunk, preserving word boundaries where possible."""
    if overlap <= 0 or not prev_chunk:
        return ""
    tail = prev_chunk[-overlap:]
    # Try to start at a word boundary (first space in the tail).
    space_idx = tail.find(" ")
    if space_idx != -1 and space_idx < len(tail) - 1:
        return tail[space_idx + 1:]
    return tail


# ===============================================================
# TABULAR NORMALIZATION HELPERS
# ===============================================================
def _to_clean_str(value: Any) -> str:
    """Return a trimmed string representation for a cell value."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _count_non_empty_cells(cells: Sequence[Any]) -> int:
    """Count non-empty cell values in a row."""
    return sum(1 for cell in cells if _to_clean_str(cell))


def _looks_numeric_cell(value: str) -> bool:
    """Return True if a cell mostly looks numeric."""
    text = value.strip()
    if not text:
        return False
    compact = text.replace(",", "").replace(" ", "")
    if compact.startswith(("+", "-")):
        compact = compact[1:]
    if compact.count(".") > 1:
        return False
    if "." in compact:
        left, right = compact.split(".", 1)
        return left.isdigit() and right.isdigit()
    return compact.isdigit()


def _score_header_candidate(
    row_cells: Sequence[Any],
    next_row_cells: Optional[Sequence[Any]] = None,
) -> float:
    """
    Score one row as a potential header.

    Higher score means "more likely to be column headers". The score
    favors rows with many non-empty, mostly textual, mostly unique
    values and common header keywords.
    """
    values = [_to_clean_str(cell) for cell in row_cells]
    non_empty = [value for value in values if value]
    if not non_empty:
        return float("-inf")

    non_empty_count = len(non_empty)
    lowered = [value.lower() for value in non_empty]
    unique_ratio = len(set(lowered)) / max(1, non_empty_count)

    alpha_count = sum(1 for value in non_empty if any(ch.isalpha() for ch in value))
    numeric_like_count = sum(1 for value in non_empty if _looks_numeric_cell(value))

    keyword_hits = 0
    for value in lowered:
        normalized = re.sub(r"[_\s]+", " ", value)
        tokens = normalized.split()
        if any(token in _HEADER_KEYWORDS for token in tokens):
            keyword_hits += 1

    continuity_bonus = 0.0
    if next_row_cells is not None:
        next_non_empty = _count_non_empty_cells(next_row_cells)
        if next_non_empty >= max(1, non_empty_count // 2):
            continuity_bonus = 1.0

    score = (
        non_empty_count * 1.5
        + alpha_count * 2.0
        + unique_ratio * 2.0
        + keyword_hits * 4.0
        - numeric_like_count * 1.5
        + continuity_bonus
    )
    return score


def _detect_header_row(
    df_raw: "pd.DataFrame",
    max_probe_rows: int = 50,
) -> int:
    """
    Detect the most likely header row index in a raw tabular dataframe.

    The dataframe is expected to be read with header=None so every row
    is available for scoring.
    """
    if df_raw.empty:
        return 0

    probe_rows = min(len(df_raw), max_probe_rows)
    best_idx = 0
    best_score = float("-inf")

    for idx in range(probe_rows):
        row_cells = list(df_raw.iloc[idx].tolist())
        next_row_cells = None
        if idx + 1 < len(df_raw):
            next_row_cells = list(df_raw.iloc[idx + 1].tolist())
        score = _score_header_candidate(row_cells, next_row_cells)
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score == float("-inf"):
        for idx in range(probe_rows):
            row_cells = list(df_raw.iloc[idx].tolist())
            if _count_non_empty_cells(row_cells) > 0:
                return idx
        return 0

    return best_idx


def _normalize_column_names(raw_columns: Sequence[Any]) -> List[str]:
    """Create clean, unique, non-empty column names."""
    names: List[str] = []
    seen: Dict[str, int] = {}

    for idx, raw_col in enumerate(raw_columns, start=1):
        name = _to_clean_str(raw_col)
        if not name or name.lower().startswith("unnamed"):
            name = f"column_{idx}"
        name = re.sub(r"\s+", " ", name)

        key = name.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            name = f"{name}_{seen[key]}"

        names.append(name)

    return names


def _drop_empty_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop rows where every cell is empty after trimming."""
    keep_mask = [
        any(_to_clean_str(value) for value in row_values)
        for row_values in df.itertuples(index=False, name=None)
    ]
    if not any(keep_mask):
        return df.iloc[0:0].copy()
    return df.loc[keep_mask].reset_index(drop=True)


def _drop_empty_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop columns where all values are empty."""
    keep_cols: List[str] = []
    for col in df.columns:
        series = df[col]
        has_value = any(_to_clean_str(value) for value in series.tolist())
        if has_value:
            keep_cols.append(col)

    if not keep_cols:
        return df.iloc[:, 0:0].copy()
    return df[keep_cols].copy()


def _normalize_for_compare(value: str) -> str:
    """Normalize strings for safe header/data equality checks."""
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _drop_duplicate_header_first_row(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Drop the first data row if it duplicates column headers.

    Some spreadsheet exports duplicate the header row once data starts.
    """
    if df.empty:
        return df

    header_values = [_normalize_for_compare(str(col)) for col in df.columns]
    first_row_values = [
        _normalize_for_compare(_to_clean_str(value))
        for value in df.iloc[0].tolist()
    ]
    if first_row_values == header_values:
        return df.iloc[1:].reset_index(drop=True)
    return df


def _prepare_dataframe_from_raw(
    df_raw: "pd.DataFrame",
    filename: str,
    header_row: Optional[int] = None,
) -> "pd.DataFrame":
    """
    Build a clean dataframe from a raw header=None table.

    Steps:
      1) Resolve header row (provided override or heuristic).
      2) Promote that row to column names.
      3) Keep rows below header.
      4) Drop empty rows/columns and duplicate first header row.
    """
    if df_raw.empty:
        raise ValueError(
            f"File '{filename}' has no tabular content after parsing."
        )

    if header_row is None:
        header_idx = _detect_header_row(df_raw)
    else:
        header_idx = int(header_row)

    if header_idx < 0 or header_idx >= len(df_raw):
        raise ValueError(
            f"header_row={header_idx} is out of range for '{filename}'."
        )

    header_cells = list(df_raw.iloc[header_idx].tolist())
    columns = _normalize_column_names(header_cells)

    body = df_raw.iloc[header_idx + 1 :].copy()
    body.columns = columns
    body = _drop_empty_rows(body)
    body = _drop_duplicate_header_first_row(body)
    body = _drop_empty_rows(body)
    body = _drop_empty_columns(body)

    if body.empty:
        raise ValueError(
            f"File '{filename}' has no data rows after header normalization."
        )
    if len(body.columns) == 0:
        raise ValueError(
            f"File '{filename}' has no non-empty columns after normalization."
        )

    return body.reset_index(drop=True)


def _serialize_tabular_row(
    row: Dict[str, str],
    columns: Sequence[str],
) -> str:
    """
    Serialize a tabular row into deterministic text for embedding.

    Format:
        ColumnA: valueA | ColumnB: valueB | ...
    """
    parts: List[str] = []
    for col in columns:
        value = _to_clean_str(row.get(col, ""))
        if not value:
            continue
        parts.append(f"{col}: {value}")
    return " | ".join(parts)


def _apply_skip_rows(
    rows: List[List[str]],
    skip_rows: Optional[int | List[int]],
    filename: str,
) -> List[List[str]]:
    """Apply skip-rows behavior to raw CSV row lists."""
    if skip_rows is None:
        return rows

    if isinstance(skip_rows, int):
        if skip_rows < 0:
            raise ValueError(
                f"skip_rows cannot be negative for file '{filename}'."
            )
        return rows[skip_rows:]

    if isinstance(skip_rows, list):
        invalid = [item for item in skip_rows if not isinstance(item, int) or item < 0]
        if invalid:
            raise ValueError(
                "skip_rows list must contain only non-negative integers."
            )
        skip_set = set(skip_rows)
        return [row for idx, row in enumerate(rows) if idx not in skip_set]

    raise ValueError(
        f"skip_rows has invalid type {type(skip_rows).__name__}; "
        "expected int, list[int], or None."
    )


# ===============================================================
# FILE EXTRACTION
# ===============================================================

def _read_text_with_fallback(file_path: str) -> str:
    """Read a file as text, trying UTF-8 first then Latin-1."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def extract_from_txt(
    file_path: str,
    filename: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract text content from a plain-text file.

    Returns a single-element list: [(full_text, metadata)].
    Chunking is applied later by the orchestrator.
    """
    content = _read_text_with_fallback(file_path).strip()
    if not content:
        raise ValueError(
            f"File '{filename}' is empty or contains only whitespace."
        )
    return [(content, {"source": filename, "file_type": "txt"})]


def extract_from_csv(
    file_path: str,
    filename: str,
    *,
    header_row: Optional[int] = None,
    skip_rows: Optional[int | List[int]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract rows from a CSV file using pandas.

    Reads the table as raw rows first (header=None), resolves the most
    likely header row, then serializes each data row into deterministic
    key/value text for embedding.
    """
    import pandas as pd

    try:
        content = _read_text_with_fallback(file_path)
    except Exception as exc:
        raise ValueError(
            f"Failed to read CSV file '{filename}': {exc}"
        ) from exc

    if not content.strip():
        raise ValueError(
            f"CSV file '{filename}' is empty or contains only whitespace."
        )

    sample_lines = content.splitlines()
    sample = "\n".join(sample_lines[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    try:
        reader = csv.reader(io.StringIO(content), dialect)
        rows = [list(row) for row in reader]
    except Exception as exc:
        raise ValueError(
            f"Failed to parse CSV file '{filename}': {exc}"
        ) from exc

    rows = _apply_skip_rows(rows, skip_rows, filename)
    if not rows:
        raise ValueError(
            f"CSV file '{filename}' has no rows after applying skip_rows."
        )

    max_cols = max(len(row) for row in rows)
    normalized_rows = [
        row + ([""] * (max_cols - len(row)))
        for row in rows
    ]
    df_raw = pd.DataFrame(normalized_rows, dtype=str)

    df = _prepare_dataframe_from_raw(
        df_raw,
        filename=filename,
        header_row=header_row,
    )
    return _extract_from_dataframe(df, filename, file_type="csv")


def extract_from_excel(
    file_path: str,
    filename: str,
    *,
    sheet_name: int | str = 0,
    header_row: Optional[int] = None,
    skip_rows: Optional[int | List[int]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract rows from an Excel file (first sheet only) using pandas.

    Reads the sheet as raw rows first (header=None), resolves the most
    likely header row, then serializes each data row into deterministic
    key/value text for embedding.
    """
    import pandas as pd

    try:
        df_raw = pd.read_excel(
            file_path,
            header=None,
            dtype=str,
            keep_default_na=False,
            sheet_name=sheet_name,
            skiprows=skip_rows,
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to parse Excel file '{filename}': {exc}"
        ) from exc

    df = _prepare_dataframe_from_raw(
        df_raw,
        filename=filename,
        header_row=header_row,
    )
    return _extract_from_dataframe(
        df,
        filename,
        file_type="excel",
        sheet_name=sheet_name,
    )


def _extract_from_dataframe(
    df: "pd.DataFrame",
    filename: str,
    file_type: str,
    sheet_name: Optional[int | str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Shared logic for CSV and Excel: serialize every data row into
    deterministic text + metadata tuples."""
    columns = list(df.columns)
    results: List[Tuple[str, Dict[str, Any]]] = []

    for logical_row, row_values in enumerate(
        df.itertuples(index=False, name=None),
        start=1,
    ):
        row_map = {
            col: _to_clean_str(value)
            for col, value in zip(columns, row_values)
        }
        text = _serialize_tabular_row(row_map, columns)
        if not text:
            continue

        meta: Dict[str, Any] = {
            "source": filename,
            "file_type": file_type,
            "row": str(logical_row),
        }
        if sheet_name is not None:
            meta["sheet"] = str(sheet_name)
        for col in columns:
            val = row_map[col]
            if val:
                meta[col] = val

        results.append((text, meta))

    if not results:
        raise ValueError(
            f"File '{filename}' produced no non-empty text rows after parsing."
        )
    return results


def _chunk_without_overlap(text: str, max_chars: int) -> List[str]:
    """Split text into <= max_chars pieces with zero overlap."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    pieces = _split_long_segment(text, max_chars)
    result: List[str] = []
    for piece in pieces:
        clean_piece = piece.strip()
        if not clean_piece:
            continue
        if len(clean_piece) <= max_chars:
            result.append(clean_piece)
            continue
        for i in range(0, len(clean_piece), max_chars):
            hard_piece = clean_piece[i : i + max_chars].strip()
            if hard_piece:
                result.append(hard_piece)
    return result


def _chunk_tabular_text(text: str, max_chars: int) -> List[str]:
    """
    Chunk one serialized CSV/Excel row with no overlap.

    Policy:
      - If the full row fits, keep one chunk.
      - Else split on field boundaries (" | ").
      - If a single field is still too long, split that field further
        by words/hard-cut while preserving zero overlap.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    fields = [field.strip() for field in text.split(" | ") if field.strip()]
    if len(fields) <= 1:
        return _chunk_without_overlap(text, max_chars)

    chunks: List[str] = []
    current = ""

    for field in fields:
        if len(field) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_chunk_without_overlap(field, max_chars))
            continue

        candidate = field if not current else f"{current} | {field}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = field

    if current:
        chunks.append(current)

    return [chunk for chunk in chunks if chunk]

# ===============================================================
# ORCHESTRATOR
# ===============================================================

def validate_file(file_path: str, filename: str) -> str:
    """
    Check that *file_path* exists, is within the size limit, and has
    a supported extension.  Returns the normalised extension (e.g. ".csv").
    """
    if not os.path.isfile(file_path):
        raise ValueError(f"File not found: {filename}")

    size = os.path.getsize(file_path)
    if size == 0:
        raise ValueError(f"File '{filename}' is empty (0 bytes).")
    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        raise ValueError(
            f"File '{filename}' is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB."
        )

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def process_file(
    file_path: str,
    original_filename: str,
    max_chars: int = 500,
    overlap: int = 50,
    *,
    header_row: Optional[int] = None,
    skip_rows: Optional[int | List[int]] = None,
    sheet_name: int | str = 0,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Full pipeline: validate → extract → chunk.

    Returns parallel lists ready for VectorStore.insert_batch():
        texts:         List of chunk strings (each <= max_chars).
        metadata_list: List of metadata dicts (one per chunk).

    Each chunk's metadata includes:
        source       – original filename
        file_type    – "txt", "csv", or "excel"
        chunk_index  – 0-based position within the parent document/row
        total_chunks – how many chunks that document/row produced
        (plus any per-row metadata extracted from CSV/Excel columns)

    Chunking policy:
        - TXT uses chunk_text() and honors overlap.
        - CSV/Excel uses row-wise chunking with zero overlap.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= max_chars:
        raise ValueError("overlap must be strictly less than max_chars.")

    ext = validate_file(file_path, original_filename)

    # --- Extract raw (text, metadata) tuples --------------------
    if ext == ".txt":
        raw = extract_from_txt(file_path, original_filename)
    elif ext == ".csv":
        raw = extract_from_csv(
            file_path,
            original_filename,
            header_row=header_row,
            skip_rows=skip_rows,
        )
    elif ext in {".xlsx", ".xls"}:
        raw = extract_from_excel(
            file_path,
            original_filename,
            sheet_name=sheet_name,
            header_row=header_row,
            skip_rows=skip_rows,
        )
    else:
        raise ValueError(f"Unsupported extension: {ext}")

    # --- Chunk each extracted text and expand metadata -----------
    all_texts: List[str] = []
    all_meta: List[Dict[str, Any]] = []

    for text, base_meta in raw:
        if ext == ".txt":
            chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
        else:
            chunks = _chunk_tabular_text(text, max_chars=max_chars)
        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            meta = dict(base_meta)
            meta["chunk_index"] = str(idx)
            meta["total_chunks"] = str(total)
            all_texts.append(chunk)
            all_meta.append(meta)

    if not all_texts:
        raise ValueError(
            f"File '{original_filename}' produced no text chunks after processing."
        )

    return all_texts, all_meta
