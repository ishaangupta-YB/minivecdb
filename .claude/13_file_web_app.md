# MiniVecDB — File: `web/` (Flask Web Interface, v3.0)

> **Location**: `minivecdb/web/`
> **Total size**: `app.py` (~680 lines) + 8 templates
> **Role**: Session-aware web UI with search, insert, file upload, history, stats, and JSON API

---

## Overview

The web module is a **fully implemented** Flask application providing a browser-based interface to MiniVecDB. It is session-aware: the home page is a session picker, and all other views are gated on an active session.

### Design Principles
- **No auto-seed**: The app starts empty; the user picks or creates a session first
- **Session-scoped**: All operations (search, insert, history, stats) are scoped to the active session
- **Query logging**: Every search and insert is logged to the `messages` table
- **Shared DB**: Uses the shared `db_run/minivecdb.db` via `ensure_shared_db_exists()`
- **Process-global VectorStore**: A single `_store` variable is rebound when the user switches sessions

---

## File: `web/app.py` (~680 lines)

### Module-Level State

```python
_store: Optional[VectorStore] = None          # Active VectorStore instance
_store_session_name: Optional[str] = None     # Name of the bound session
```

### Helper Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_bind_store(session_folder_abs_path)` | 59-74 | Closes current store, opens new one, sets `.active_run` marker |
| `_active_store()` | 77-94 | Returns active store; auto-rebinds from `.active_run` if needed |
| `_format_time(ts)` | 97-103 | Unix timestamp → `"YYYY-MM-DD HH:MM:SS"` display string |
| `_format_score(score, metric)` | 106-109 | Cosine → percentage, others → 4-decimal float |
| `_list_sessions_dicts()` | 111-142 | Opens throwaway DB, queries `list_sessions`, filters out `__picker__` |

### Flask Routes

| Route | Method | Handler | Template | Purpose |
|-------|--------|---------|----------|---------|
| `GET /` | GET | `index()` | `select_session.html` | Session picker landing page |
| `POST /session/new` | POST | `session_new()` | redirect → `/search-page` | Create new session + bind |
| `POST /session/switch` | POST | `session_switch()` | redirect → `/search-page` | Bind existing session |
| `GET /search-page` | GET | `search_page()` | `index.html` | Search form with metadata filter (key + value) |
| `POST /search` | POST/GET | `search()` | `results.html` | Execute search + log + show results |
| `GET /stats` | GET | `stats()` | `stats.html` | Per-session database statistics |
| `GET/POST /insert` | GET/POST | `insert()` | `insert.html` | Insert page: "Paste text" tab + "Upload file" tab |
| `GET/POST /upload` | GET/POST | `upload()` | `insert.html` | File upload handler; renders insert page with file tab active |
| `GET /records` | GET | `records()` | `records.html` | Paginated record browser (optional collection filter) |
| `GET /history` | GET | `history()` | `history.html` | Chat history timeline |
| `GET /api/search` | GET | `api_search()` | JSON response | JSON API endpoint |
| `GET /favicon.ico` | GET | `favicon()` | 204 No Content | Suppress favicon 404s |

### Route Details

#### `GET /` — Session Picker
- Calls `_list_sessions_dicts()` to get all sessions with counts
- Reads `.active_run` to highlight currently active session
- Renders `select_session.html` with session list

#### `POST /session/new` — New Session
- Creates a fresh run folder via `create_new_run_path(prefix="demo")`
- Binds the VectorStore to it
- Redirects to `/search-page`

#### `POST /session/switch` — Switch Session
- Reads `session_name` from form data
- Validates the path exists and is inside `db_run/`
- Binds VectorStore and redirects

#### `POST /search` — Execute Search
1. Extracts `query`, `metric`, `top_k`, `filter_key`, `filter_value` from form
2. Clamps `top_k` to [1, 50]
3. Builds `filters = {filter_key: filter_value}` only when both are non-empty
4. Calls `store.search()` with timing
5. Logs a `messages` row with `kind='search'` and `category_filter="key:value"`
6. Handles `ValueError` as user-facing input/search errors
7. Handles unexpected exceptions with safe error text (and server logs)
8. Renders results with score display

#### `GET/POST /insert` — Insert Page (Two Tabs)
1. GET: Shows insert page with two tabs: "Paste text" and "Upload file"
2. POST (text tab): Extracts `text` + up to 3 metadata key-value pairs
3. Calls `store.insert()` with timing
4. Logs a `messages` row with `kind='insert'` and `response_ref=new_id`
5. Shows success message with new record ID

#### `GET/POST /upload` — File Upload Handler
1. GET: Redirects to insert page with the "Upload file" tab active
2. POST: Saves uploaded file to temp, calls `process_file()` from `core.file_processor`:
   - validates ≤10 MB
   - TXT: semantic chunking with overlap
   - CSV/Excel: robust header normalization + row-first chunking with zero overlap
   then calls `store.insert_batch()`
3. Logs to messages with `kind='insert'` and `query_text='[file upload] <filename>'`
4. Renders insert page (file tab) with success summary: filename, chunk count, collection, elapsed time, sample IDs

#### `GET /api/search` — JSON API
- Query params: `q` (required), `metric` (default cosine), `top_k` (default 5), `filter_key`, `filter_value` (also supports legacy `category` param for backwards compat)
- Returns JSON payload: `{session, query, metric, top_k, filter_key, filter_value, elapsed_ms, count, results}`
- Error responses: 400 (missing/bad params), 409 (no session), 500 (unexpected runtime failure)
- Logs to messages table same as POST /search

### Entry Point

```python
def main() -> None:
    ensure_shared_db_exists(ensure_db_run_root())
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
```

Run with: `python -m web.app` → http://localhost:5000

---

## Templates (8 Files)

All templates extend `_base.html` via Jinja2 inheritance.

### `_base.html` (405 lines, 9.8 KB) — Shared Layout

**Sections**:
- **CSS Design System** (lines 7-356): Complete design system with CSS custom properties:
  - Color palette: `--bg`, `--text-main`, `--primary`, `--accent`, `--success-*`, `--error-*`
  - Typography: Inter + JetBrains Mono via Google Fonts
  - Components: `.card`, `.pill`, `.btn`, `.error`, `.ok`, `.meta`, `.row`
  - Responsive: Mobile breakpoint at 560px and 768px
  - Transitions: `cubic-bezier(0.4, 0, 0.2, 1)` for smooth hover effects
- **Header**: SVG database icon + "MiniVecDB" branding + subtitle
- **Navigation**: Conditionally rendered based on `active_session`:
  - With session: Search, Insert, History, Stats links
  - Without session: "Pick a session" link
- **Session banner**: Shows `active_session` name in header with "Switch" link
- **Footer**: Links to Stats and API endpoint

### `select_session.html` (73 lines, 2.5 KB) — Session Picker

- "Start new session" button → `POST /session/new`
- Resume dropdown with record counts and last-used date → `POST /session/switch`
- Sessions table showing name, records, messages, created, last used
- Highlights currently active session with green `.ok` banner

### `index.html` (71 lines, 2.2 KB) — Search Form

- Session stats banner (records count, collections count)
- Empty-session notice with link to Insert
- Search form: query input, metric dropdown (cosine/euclidean/dot), top_k number input, metadata filter key + value inputs
- Filter accepts any metadata key-value pair (e.g. category:Science, source:sample, book:harry)

### `results.html` (96 lines, 3.5 KB) — Search Results

- Query summary line: query text, metric, top_k, metadata filter (key:value), elapsed time
- Re-search form with pre-filled values for iterative refinement
- Result cards: rank badge (#1, #2...), score display (percentage for cosine), full text, metadata pills, record ID
- "No results" fallback when empty

### `insert.html` (~180 lines) — Two-Tab Insert + Upload UI

- Tab 1: Paste text + metadata key/value rows
- Tab 2: Upload file (`.txt`, `.csv`, `.xlsx`, `.xls`) + collection picker + extra metadata
- Upload success panel with chunk count, elapsed time, sample IDs
- Error/success banners shared across both tabs

### `records.html` — Record Browser

- Paginated table view of records in the active session
- Optional collection filter
- Shows text preview, metadata summary, created timestamp

### `history.html` (42 lines, 1.5 KB) — Chat History Timeline

- Scrollable table: When, Kind (pill), Query, Metric, Top K, Filter, Results, ms, Ref
- Shows all messages in chronological order for the active session
- Empty state message when no history

### `stats.html` (49 lines, 1.6 KB) — Database Statistics

- Summary table: total records, collections, vector dimension, memory (bytes + MB), embedding model, storage path, SQLite file
- Collections table: name, dimension, records, description

---

## Template Hierarchy

```
_base.html
├── select_session.html   (GET /)
├── index.html            (GET /search-page)
├── results.html          (POST /search)
├── insert.html           (GET/POST /insert + /upload)
├── records.html          (GET /records)
├── history.html          (GET /history)
└── stats.html            (GET /stats)
```

---

## Security & Design Notes

1. **No authentication**: This is a learning project, not production. No login system.
2. **Session validation**: `session_switch` validates the path exists AND is inside `db_run/` to prevent directory traversal.
3. **Logging never breaks operations**: `log_message` calls are wrapped in bare `except Exception: pass` so a logging failure never breaks a search or insert.
4. **`__picker__` session**: The session picker creates a throwaway `DatabaseManager` bound to a sentinel session `"__picker__"` to access `list_sessions_with_counts`. This session is filtered out of the displayed list.
5. **top_k clamping**: Web UI clamps to [1, 50]; API clamps to [1, 100].
