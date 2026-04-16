# 📚 MiniVecDB — Complete Codebase Documentation

> **Last Updated**: April 16, 2026
> **Project**: MiniVecDB — A mini vector database built from scratch in Python
> **Course**: University DBMS (Database Management Systems)

## Documentation Index

This folder contains **15 documentation files** that exhaustively explain the MiniVecDB codebase. Each file covers a specific aspect of the project.

### 🗺️ High-Level Documents

| # | File | Covers | Key Topics |
|---|------|--------|------------|
| 01 | [01_overview.md](./01_overview.md) | Full project overview | What, why, core concepts, tech stack, project structure |
| 02 | [02_architecture_and_flow.md](./02_architecture_and_flow.md) | Architecture & data flow | Layer diagram, INSERT/SEARCH/GET/DELETE flows, persistence model, three-way bridge |
| 03 | [03_er_diagram_and_schema.md](./03_er_diagram_and_schema.md) | ER diagram & SQL schema (v3.0) | 6 tables, 3 triggers, 9 indexes, cascade deletes, SQL technique catalog |

### 📄 File-by-File Deep Dives

| # | File | Source File | Lines | What You'll Learn |
|---|------|------------|-------|-------------------|
| 04 | [04_file_ARCHITECTURE.md](./04_file_ARCHITECTURE.md) | `ARCHITECTURE.py` | 526 | Central spec: 6-table schema, 35 SQL queries, 7 data models, generate_id, self-test |
| 05 | [05_file_distance_metrics.md](./05_file_distance_metrics.md) | `core/distance_metrics.py` | 519 | 3 metrics (cosine, euclidean, dot), single + batch variants, mathematical formulas |
| 06 | [06_file_embeddings.md](./06_file_embeddings.md) | `core/embeddings.py` | 568 | EmbeddingEngine (neural), SimpleEmbeddingEngine (BoW), factory pattern |
| 07 | [07_file_runtime_paths.md](./07_file_runtime_paths.md) | `core/runtime_paths.py` | 233 | 15 functions: path resolution, active run, run creation, shared DB path |
| 08 | [08_file_vector_store.md](./08_file_vector_store.md) | `core/vector_store.py` | 1705 | ★ The heart: CRUD, search, collections, persistence, shared DB routing |
| 09 | [09_file_database.md](./09_file_database.md) | `storage/database.py` | 565 | Session-bound SQLite wrapper: CRUD, metadata filters, sessions/messages |
| 10 | [10_file_cli.md](./10_file_cli.md) | `cli/main.py` | ~805 | All 10 CLI commands, argparse setup, output formatting |
| 11 | [11_supporting_files_and_tests.md](./11_supporting_files_and_tests.md) | Various | — | Package inits, requirements, test suite (16 files), disk layout, config |
| 12 | [12_file_data_benchmarks_demo.md](./12_file_data_benchmarks_demo.md) | `data/`, `tests/benchmark.py`, `demo/` | 1175 | Curated 150+ doc dataset, benchmarks (both engines), demo app, results |
| 13 | [13_file_web_app.md](./13_file_web_app.md) | `web/app.py` + 8 templates | ~680+450 | Flask web UI: 12 routes, session picker, search/insert/upload/history/stats, JSON API |
| 14 | [14_file_migrations.md](./14_file_migrations.md) | `storage/migrations.py` | 276 | Legacy migration: per-session → shared DB, collision handling, safety guarantees |
| 15 | [15_file_upload.md](./15_file_upload.md) | `core/file_processor.py` | ~850 | File upload pipeline: robust tabular normalization + format-specific chunking |

---

## Recommended Reading Order

1. **Start here**: `01_overview.md` → understand what and why
2. **Architecture**: `02_architecture_and_flow.md` → how pieces connect
3. **Schema**: `03_er_diagram_and_schema.md` → database design
4. **Central spec**: `04_file_ARCHITECTURE.md` → data models and SQL
5. **Search engine**: `05_file_distance_metrics.md` → the math
6. **Embeddings**: `06_file_embeddings.md` → text → vector
7. **Main engine**: `08_file_vector_store.md` → the heart
8. **Database**: `09_file_database.md` → SQLite access layer
9. **Migrations**: `14_file_migrations.md` → legacy DB migration
10. **Web**: `13_file_web_app.md` → Flask interface
11. **CLI**: `10_file_cli.md` → terminal interface
12. **Data & benchmarks**: `12_file_data_benchmarks_demo.md` → dataset + perf testing

---

## Quick System Reference

| System | Technology | File(s) |
|--------|-----------|---------|
| Vector math | NumPy | `core/distance_metrics.py` |
| Embeddings | sentence-transformers | `core/embeddings.py` |
| Structured storage | SQLite3 (v3.0: shared DB) | `storage/database.py`, `ARCHITECTURE.py` |
| File upload/chunking | csv + pandas + openpyxl | `core/file_processor.py` |
| Legacy migration | sqlite3 | `storage/migrations.py` |
| Path management | OS file system | `core/runtime_paths.py` |
| Core engine | Python | `core/vector_store.py` |
| CLI | argparse | `cli/main.py` |
| Web | Flask + Jinja2 | `web/app.py`, `web/templates/` |
| Testing | pytest + standalone | `tests/` (run via `.venv/bin/python -m pytest tests/ -v` or `.venv/bin/python tests/run_all_tests.py`) |
| Benchmarking | NumPy + both engines | `tests/benchmark.py` |
| Demo dataset | JSON shards | `data/generated/` |

---

## Source Code Statistics

| File | Lines | Size | Role |
|------|-------|------|------|
| `ARCHITECTURE.py` | 526 | 21 KB | Central spec (v3.0) |
| `core/distance_metrics.py` | 519 | 22 KB | Similarity math |
| `core/embeddings.py` | 568 | 25 KB | Text → vectors |
| `core/runtime_paths.py` | 233 | 8 KB | Path management |
| `core/file_processor.py` | ~850 | ~31 KB | Upload extraction + chunking pipeline |
| `core/vector_store.py` | 1705 | 66 KB | Main engine |
| `storage/database.py` | 565 | 22 KB | SQLite wrapper (v3.0) |
| `storage/migrations.py` | 276 | 9.6 KB | Legacy migration |
| `cli/main.py` | ~805 | ~26 KB | CLI interface |
| `web/app.py` | ~680 | ~22 KB | Flask web app |
| `web/templates/` (8 files) | ~970 | ~30 KB | Jinja2 templates |
| `data/sample_dataset.py` | 59 | 1.6 KB | Dataset loader |
| `data/generated/*.json` | — | 51 KB | 150+ curated docs |
| `demo/semantic_search.py` | 353 | 15 KB | End-to-end demo |
| `tests/benchmark.py` | 1798 | 67 KB | Performance benchmarks |
| **Total (source)** | **~8645** | **~372 KB** | |
| Test files (17 files) | ~4000 | ~230 KB | Test suite + benchmarks |
| **Grand Total** | **~12,300** | **~582 KB** | |
