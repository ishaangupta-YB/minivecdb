# MiniVecDB — Quick Reference Guide
# ===================================
# This tells you exactly what to do, step by step.


## WHAT YOU RECEIVED (8 files)

| # | File                              | What it is                                             | Where to put it                  |
|---|-----------------------------------|--------------------------------------------------------|----------------------------------|
| 1 | `CLAUDE.md`                       | Master prompt — Claude Code reads this automatically   | Project root: `minivecdb/`       |
| 2 | `ARCHITECTURE.py`                 | Data models, SQL schema, design spec (updated w/ SQLite)| Project root: `minivecdb/`       |
| 3 | `DAY_PROMPTS.py`                  | All 16 day-by-day prompts (Days 5-20)                  | Keep anywhere for reference      |
| 4 | `setup_project.py`                | Creates the folder structure automatically             | Run once, then discard           |
| 5 | `day4_full_architecture_v2.html`  | Complete architecture diagram (SQLite + NumPy)         | For your report/reference        |
| 6 | `day4_er_diagram.html`            | SQLite entity-relationship diagram                     | For your report/reference        |
| 7 | `day4_class_design.html`          | VectorStore class design diagram                       | For your report/reference        |
| 8 | `hybrid_architecture_sql_numpy.html` | What goes in SQLite vs NumPy diagram                | For your report/reference        |


## FILES YOU ALREADY HAVE (from Days 1-3 in this chat)

| File                              | What it is                          |
|-----------------------------------|-------------------------------------|
| `core/distance_metrics.py`        | 3 similarity metrics — DONE ✅      |
| `core/embeddings.py`              | Text-to-vector engine — DONE ✅     |
| `tests/test_distance_metrics.py`  | Pytest tests (46 tests) — DONE ✅   |
| `tests/run_tests_distance.py`     | Standalone test runner — DONE ✅     |
| `tests/run_tests_embeddings.py`   | Embeddings tests (17 tests) — DONE ✅|
| `MiniVecDB_Project_Proposal.docx` | Project proposal document — DONE ✅  |


## SETUP STEPS (do this once)

```bash
# Step 1: Run the setup script to create folder structure
python setup_project.py minivecdb

# Step 2: Place files in the right locations
cp CLAUDE.md minivecdb/
cp ARCHITECTURE.py minivecdb/

# Step 3: Copy your existing Day 1-3 files into the structure
cp distance_metrics.py minivecdb/core/
cp embeddings.py minivecdb/core/
cp run_tests_distance.py minivecdb/tests/
cp run_tests_embeddings.py minivecdb/tests/
cp test_distance_metrics.py minivecdb/tests/

# Step 4: Install dependencies
cd minivecdb
pip install -r requirements.txt

# Step 5: Verify everything works
python ARCHITECTURE.py                    # Should say "ALL VALIDATIONS PASSED"
python tests/run_tests_distance.py        # Should say "46 passed, 0 failed"
python tests/run_tests_embeddings.py      # Should say "17 passed, 0 failed"
```


## DAILY WORKFLOW (repeat for Days 5-20)

```
1. Open terminal, cd into minivecdb/
2. Start Claude Code (or open it in your IDE)
3. Open DAY_PROMPTS.py, find the prompt for today's day
4. Copy ONLY that day's prompt text (the string inside the triple quotes)
5. Paste it into Claude Code
6. Let the agent write the code — review it, ask questions
7. Run the tests the agent created
8. If tests pass → move to next day tomorrow
9. If tests fail → tell the agent what failed and ask it to fix
```


## WHAT GETS BUILT EACH DAY

```
Day  │ What gets built                              │ Key files created
─────┼────────────────────────────────────────────────┼──────────────────────────────────
  5  │ SQLite wrapper + VectorStore (insert/get)     │ storage/database.py, core/vector_store.py
  6  │ Search engine (similarity search)             │ vector_store.py (search method added)
  7  │ Delete, Update, Collections                   │ vector_store.py (CRUD completed)
  8  │ Save/Load persistence                         │ vector_store.py (save/load perfected)
  9  │ Integration + edge case tests                 │ tests/test_integration.py
 10  │ Advanced metadata filtering ($gt, $lt, etc)   │ storage/database.py (enhanced)
 11  │ Command-line interface                        │ cli/main.py
 12  │ Simple partition index (for benchmarks)       │ core/indexing.py
 13  │ Performance benchmarking                      │ tests/benchmark.py
 14  │ Dataset creation + demo setup                 │ data/sample_dataset.py, demo/semantic_search.py
 15  │ Flask web interface                           │ web/app.py, web/templates/
 16  │ Error handling + polish                       │ All files refined
 17  │ Comprehensive test suite                      │ tests/test_comprehensive.py
 18  │ Project report                                │ docs/REPORT.md
 19  │ Presentation prep                             │ docs/PRESENTATION_OUTLINE.md
 20  │ Final review + submission package             │ Everything cleaned and zipped
```


## IF SOMETHING BREAKS

Tell the agent exactly what happened:
```
"The test test_search_returns_ranked_results is failing with this error:
[paste the full error traceback here]
Fix this before we proceed."
```

The agent will read CLAUDE.md automatically and have full project context.


## IMPORTANT REMINDERS

- Only give ONE day's prompt at a time — don't combine days
- Always verify previous day's tests pass before starting the next day
- The CLAUDE.md file must stay in the project root — Claude Code reads it automatically
- If the agent tries to install ChromaDB, FAISS, or any vector DB library → STOP IT
  The whole point is building the vector search from scratch
- If the agent skips writing comments/explanations → remind it: "I'm learning,
  explain every function with detailed comments"
- Keep DAY_PROMPTS.py open as reference but never paste the entire file at once