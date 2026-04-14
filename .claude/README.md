# 📚 MiniVecDB — Complete Codebase Documentation

> **Generated**: April 13, 2026
> **Project**: MiniVecDB — A mini vector database built from scratch in Python
> **Course**: University DBMS (Database Management Systems)

---

## Documentation Index

This folder contains 11 documentation files that exhaustively explain the MiniVecDB codebase. Each file covers a specific aspect of the project.

### 🗺️ High-Level Documents

| # | File | What It Covers |
|---|------|---------------|
| 01 | [01_overview.md](./01_overview.md) | **What is MiniVecDB?** Core concepts (embeddings, similarity search, hybrid storage), technology stack, project structure, key design decisions |
| 02 | [02_architecture_and_flow.md](./02_architecture_and_flow.md) | **How does it work?** Layered architecture, data flow diagrams for INSERT/SEARCH/GET/DELETE, startup flow, persistence model, crash recovery, the three-way bridge |
| 03 | [03_er_diagram_and_schema.md](./03_er_diagram_and_schema.md) | **Database design** — ER diagram, all 3 SQLite tables, relationships, cascade deletes, indexes, every SQL query documented |

### 📁 File-by-File Deep Dives

| # | File | Source File | Lines | What It Covers |
|---|------|------------|-------|---------------|
| 04 | [04_file_ARCHITECTURE.md](./04_file_ARCHITECTURE.md) | `ARCHITECTURE.py` | 152 | Data models (VectorRecord, SearchResult, etc.), SQL schema, query templates, ID generator |
| 05 | [05_file_distance_metrics.md](./05_file_distance_metrics.md) | `core/distance_metrics.py` | 519 | All 3 metrics (cosine, euclidean, dot), single + batch versions, registry pattern, normalization |
| 06 | [06_file_embeddings.md](./06_file_embeddings.md) | `core/embeddings.py` | 568 | EmbeddingEngine (neural), SimpleEmbeddingEngine (fallback), factory function, lazy loading |
| 07 | [07_file_runtime_paths.md](./07_file_runtime_paths.md) | `core/runtime_paths.py` | 199 | Path resolution, active run tracking, unique naming, model cache location |
| 08 | [08_file_vector_store.md](./08_file_vector_store.md) | `core/vector_store.py` | 1769 | **The heart** — CRUD, search algorithm, persistence, recovery, collections, context manager |
| 09 | [09_file_database.md](./09_file_database.md) | `storage/database.py` | 786 | SQLite wrapper (Repository pattern), transactions, metadata filtering with operators |
| 10 | [10_file_cli.md](./10_file_cli.md) | `cli/main.py` | 719 | All 10 CLI commands, argparse setup, output formatting |
| 11 | [11_supporting_files_and_tests.md](./11_supporting_files_and_tests.md) | Various | — | Package inits, requirements, test suite, config files, runtime disk layout |

---

## Excalidraw Diagrams

Three interactive diagrams were created using the Excalidraw MCP tool:

1. **ER Diagram** — Shows the 3 SQLite tables (collections, records, metadata) with their fields, primary keys, foreign keys, and cascade relationships

2. **System Architecture** — Shows the 4-layer architecture: User Interface (CLI/Web/API) → Core Engine (VectorStore) → Storage (SQLite/NumPy/Embedding) → Disk files

3. **Search Flow** — Illustrates the search pipeline: Query → Embed → Pre-Filter (SQL) → Batch Similarity (NumPy) → Sort → Top-K results

---

## Quick Reference: The System in One Paragraph

MiniVecDB converts text into 384-dimensional vectors using a neural model (`all-MiniLM-L6-v2`), stores the vectors in a NumPy matrix (`vectors.npy`) and the structured data in SQLite (`minivecdb.db`), links them via a JSON bridge file (`id_mapping.json`), and finds similar documents by computing cosine similarity between the query vector and all stored vectors using fast NumPy batch operations. The `VectorStore` class coordinates everything, with the CLI and Web UI as thin interface layers on top.

---

## Reading Order

**If you want to understand the system**:
1. Start with `01_overview.md` — understand what and why
2. Read `02_architecture_and_flow.md` — understand how
3. Read `03_er_diagram_and_schema.md` — understand the data model

**If you want to understand specific code**:
- Jump directly to the file-by-file document (04–11) for the file you're looking at

**If you want to understand the search algorithm specifically**:
1. `05_file_distance_metrics.md` — the math
2. `08_file_vector_store.md` → Search Operations section — the algorithm
3. `09_file_database.md` → Metadata Operations section — the pre-filtering

---

## Source Code Statistics

| File | Lines | Size | Role |
|------|-------|------|------|
| `ARCHITECTURE.py` | 152 | 9 KB | Central spec |
| `core/distance_metrics.py` | 519 | 22 KB | Similarity math |
| `core/embeddings.py` | 568 | 23 KB | Text → vectors |
| `core/runtime_paths.py` | 199 | 7 KB | Path management |
| `core/vector_store.py` | 1769 | 68 KB | Main engine |
| `storage/database.py` | 786 | 30 KB | SQLite wrapper |
| `cli/main.py` | 719 | 23 KB | CLI interface |
| **Total (core)** | **4712** | **182 KB** | |
| Test files (13 files) | ~2500 | ~150 KB | Test suite |
| **Grand Total** | **~7200** | **~332 KB** | |
