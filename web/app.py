"""
+===============================================================+
|  MiniVecDB -- Flask Web Interface (v3.0)                      |
|  File: minivecdb/web/app.py                                   |
|                                                               |
|  Unified session-aware web UI. The home page is the session   |
|  picker: the user either resumes an existing session or       |
|  creates a new one. Only after a session is bound do search / |
|  insert / stats / history views become available.             |
|                                                               |
|  Every user query (search or insert) is logged to the shared  |
|  db_run/minivecdb.db in the messages table. No auto-seed.     |
|                                                               |
|  Run:   python -m web.app                                     |
|  Then:  http://localhost:5000                                 |
+===============================================================+
"""

import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

from core.runtime_paths import (
    ensure_db_run_root,
    get_shared_db_path,
    is_within_db_run,
    read_active_run_path,
    set_active_run_path,
)
from core.vector_store import VectorStore
from storage.database import DatabaseManager


# ---------------------------------------------------------------
# Process-wide VectorStore. Rebound (closed + reopened) whenever
# the user switches to another session via the picker.
# ---------------------------------------------------------------
_store: Optional[VectorStore] = None
_store_session_name: Optional[str] = None


def _bind_store(session_folder_abs_path: str) -> VectorStore:
    """Close the current VectorStore (if any) and open a new one."""
    global _store, _store_session_name

    if _store is not None:
        try:
            _store.close()
        except Exception:
            pass
        _store = None
        _store_session_name = None

    set_active_run_path(session_folder_abs_path)
    _store = VectorStore(storage_path=session_folder_abs_path)
    _store_session_name = _store.session_name
    return _store


def _active_store() -> Optional[VectorStore]:
    """Return the currently-bound VectorStore, rebinding from .active_run if needed."""
    global _store, _store_session_name

    active = read_active_run_path()
    if active is None:
        if _store is not None:
            try:
                _store.close()
            except Exception:
                pass
            _store = None
            _store_session_name = None
        return None

    if _store is None or os.path.abspath(_store.storage_path) != os.path.abspath(active):
        return _bind_store(active)
    return _store


def _format_time(ts: Optional[float]) -> str:
    if not ts:
        return "-"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return "-"


def _format_score(score: float, metric: str) -> str:
    if metric == "cosine":
        return f"{score * 100:.2f}%"
    return f"{score:.4f}"


def _get_categories(store: VectorStore) -> List[str]:
    """Distinct 'category' metadata values scoped to the active session."""
    rows = store.db._conn.execute(
        """
                SELECT DISTINCT TRIM(m.value)
          FROM metadata m
          JOIN records r ON r.id = m.record_id
                 WHERE r.session_id = ?
                     AND LOWER(TRIM(m.key)) = 'category'
                     AND TRIM(m.value) != ''
                 ORDER BY TRIM(m.value)
        """,
                (store.db.session_id,),
    ).fetchall()
    return [row[0] for row in rows]


def _list_sessions_dicts() -> List[Dict[str, Any]]:
    """Open the shared DB read-only and return every session with aggregates."""
    db_run_root = ensure_db_run_root()
    shared_db_path = get_shared_db_path()

    # A throwaway DatabaseManager is the cheapest way to hit the list query;
    # bind it to a neutral "__picker__" session that never gets written to.
    tmp = DatabaseManager(
        shared_db_path,
        session_name="__picker__",
        session_storage_path=db_run_root,
    )
    try:
        sessions = tmp.list_sessions()
    finally:
        tmp.close()

    visible = [s for s in sessions if s.name != "__picker__"]
    return [
        {
            "id": s.id,
            "name": s.name,
            "storage_path": s.storage_path,
            "created_at": s.created_at,
            "created_at_display": _format_time(s.created_at),
            "last_used_at": s.last_used_at,
            "last_used_display": _format_time(s.last_used_at),
            "msg_count": s.msg_count,
            "record_count": s.record_count,
        }
        for s in visible
    ]


# ---------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)

    # ==========================================================
    # GET /  -- session picker landing page
    # ==========================================================
    @app.route("/", methods=["GET"])
    def index():
        sessions = _list_sessions_dicts()
        active_path = read_active_run_path()
        active_name = os.path.basename(active_path) if active_path else None
        return render_template(
            "select_session.html",
            sessions=sessions,
            active_name=active_name,
        )

    # ==========================================================
    # POST /session/new    -- create a new session + bind
    # POST /session/switch -- bind an existing session
    # ==========================================================
    @app.route("/session/new", methods=["POST"])
    def session_new():
        from core.runtime_paths import create_new_run_path

        new_path = create_new_run_path(prefix="demo")
        _bind_store(new_path)
        return redirect(url_for("search_page"))

    @app.route("/session/switch", methods=["POST"])
    def session_switch():
        target = (request.form.get("session_name") or "").strip()
        if not target:
            return redirect(url_for("index"))

        db_run_root = ensure_db_run_root()
        candidate = os.path.abspath(os.path.join(db_run_root, target))
        if not os.path.isdir(candidate) or not is_within_db_run(candidate):
            return redirect(url_for("index"))

        _bind_store(candidate)
        return redirect(url_for("search_page"))

    # ==========================================================
    # GET /search-page  -- the search form (formerly /)
    # ==========================================================
    @app.route("/search-page", methods=["GET"])
    def search_page():
        store = _active_store()
        if store is None:
            return redirect(url_for("index"))
        return render_template(
            "index.html",
            stats=store.stats(),
            active_session=store.session_name,
        )

    # ==========================================================
    # POST /search  -- run a search, log it, render results
    # ==========================================================
    @app.route("/search", methods=["POST", "GET"])
    def search():
        store = _active_store()
        if store is None:
            return redirect(url_for("index"))

        src = request.form if request.method == "POST" else request.args
        query = (src.get("query") or "").strip()
        metric = (src.get("metric") or "cosine").strip().lower()
        category = (src.get("category") or "").strip()

        try:
            top_k = int(src.get("top_k") or 5)
        except (TypeError, ValueError):
            top_k = 5
        top_k = max(1, min(top_k, 50))

        if not query:
            return render_template(
                "results.html",
                query="",
                metric=metric,
                top_k=top_k,
                category=category,
                results=[],
                elapsed_ms=0.0,
                error="Please enter a search query.",
                format_score=_format_score,
                active_session=store.session_name,
            )

        filters: Optional[Dict[str, Any]] = None
        if category:
            filters = {"category": category}

        chosen_metric = metric if metric in {"cosine", "euclidean", "dot"} else "cosine"
        t0 = time.time()
        try:
            raw_results = store.search(
                query=query, top_k=top_k, metric=chosen_metric, filters=filters,
            )
            error = None
        except ValueError as exc:
            raw_results = []
            error = str(exc)
        elapsed_ms = (time.time() - t0) * 1000.0

        try:
            store.db.log_message(
                kind="search",
                query_text=query,
                metric=chosen_metric,
                top_k=top_k,
                category_filter=category or None,
                result_count=len(raw_results),
                elapsed_ms=round(elapsed_ms, 3),
            )
        except Exception:
            pass  # history logging must never break a search

        results = [
            {
                "rank": r.rank,
                "score": r.score,
                "score_display": _format_score(r.score, r.metric),
                "metric": r.metric,
                "text": r.record.text,
                "metadata": r.record.metadata,
                "id": r.record.id,
            }
            for r in raw_results
        ]

        return render_template(
            "results.html",
            query=query,
            metric=chosen_metric,
            top_k=top_k,
            category=category,
            results=results,
            elapsed_ms=elapsed_ms,
            error=error,
            format_score=_format_score,
            active_session=store.session_name,
        )

    # ==========================================================
    # GET /stats
    # ==========================================================
    @app.route("/stats", methods=["GET"])
    def stats():
        store = _active_store()
        if store is None:
            return redirect(url_for("index"))
        db_stats = store.stats()
        collections = store.list_collections()
        return render_template(
            "stats.html",
            stats=db_stats,
            collections=collections,
            memory_mb=db_stats.memory_usage_bytes / (1024 * 1024),
            active_session=store.session_name,
        )

    # ==========================================================
    # GET/POST /insert
    # ==========================================================
    @app.route("/insert", methods=["GET", "POST"])
    def insert():
        store = _active_store()
        if store is None:
            return redirect(url_for("index"))

        if request.method == "GET":
            return render_template(
                "insert.html",
                inserted_id=None,
                error=None,
                text="",
                active_session=store.session_name,
            )

        text = (request.form.get("text") or "").strip()
        if not text:
            return render_template(
                "insert.html",
                inserted_id=None,
                error="Text is required.",
                text="",
                active_session=store.session_name,
            )

        metadata: Dict[str, Any] = {}
        keys = request.form.getlist("meta_key")
        values = request.form.getlist("meta_value")
        for k, v in zip(keys, values):
            k = (k or "").strip()
            v = (v or "").strip()
            if k and v:
                metadata[k] = v

        t0 = time.time()
        try:
            new_id = store.insert(text=text, metadata=metadata or None)
            error = None
        except ValueError as exc:
            new_id = None
            error = str(exc)
        elapsed_ms = (time.time() - t0) * 1000.0

        try:
            store.db.log_message(
                kind="insert",
                query_text=text,
                result_count=1 if new_id else 0,
                elapsed_ms=round(elapsed_ms, 3),
                response_ref=new_id,
            )
        except Exception:
            pass

        return render_template(
            "insert.html",
            inserted_id=new_id,
            error=error,
            text=text if error else "",
            metadata=metadata,
            active_session=store.session_name,
        )

    # ==========================================================
    # GET /history  -- chronological message timeline
    # ==========================================================
    @app.route("/history", methods=["GET"])
    def history():
        store = _active_store()
        if store is None:
            return redirect(url_for("index"))

        rows = store.db.get_history(limit=500)
        formatted = [
            {
                "id": m.id,
                "created_at_display": _format_time(m.created_at),
                "kind": m.kind,
                "query_text": m.query_text,
                "metric": m.metric,
                "top_k": m.top_k,
                "category_filter": m.category_filter,
                "result_count": m.result_count,
                "elapsed_ms": m.elapsed_ms,
                "response_ref": m.response_ref,
            }
            for m in rows
        ]
        return render_template(
            "history.html",
            messages=formatted,
            active_session=store.session_name,
        )

    # ==========================================================
    # GET /api/search
    # ==========================================================
    @app.route("/api/search", methods=["GET"])
    def api_search():
        store = _active_store()
        if store is None:
            return jsonify({"error": "No active session. Open / to pick one."}), 409

        query = (request.args.get("q") or "").strip()
        if not query:
            return jsonify({"error": "Missing required query parameter 'q'."}), 400

        metric = (request.args.get("metric") or "cosine").strip().lower()
        if metric not in {"cosine", "euclidean", "dot"}:
            return jsonify({"error": f"Unknown metric '{metric}'."}), 400

        try:
            top_k = int(request.args.get("top_k") or 5)
        except (TypeError, ValueError):
            return jsonify({"error": "top_k must be an integer."}), 400
        top_k = max(1, min(top_k, 100))

        category = (request.args.get("category") or "").strip()
        filters = {"category": category} if category else None

        t0 = time.time()
        try:
            results = store.search(
                query=query, top_k=top_k, metric=metric, filters=filters,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        elapsed_ms = (time.time() - t0) * 1000.0

        try:
            store.db.log_message(
                kind="search",
                query_text=query,
                metric=metric,
                top_k=top_k,
                category_filter=category or None,
                result_count=len(results),
                elapsed_ms=round(elapsed_ms, 3),
            )
        except Exception:
            pass

        payload = {
            "session": store.session_name,
            "query": query,
            "metric": metric,
            "top_k": top_k,
            "category": category or None,
            "elapsed_ms": round(elapsed_ms, 3),
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }
        return jsonify(payload)

    @app.route("/favicon.ico")
    def favicon():
        return ("", 204)

    return app


# ---------------------------------------------------------------
# Entry point: `python -m web.app`
# ---------------------------------------------------------------
def main() -> None:
    """Start the Flask dev server without auto-seeding any dataset."""
    get_shared_db_path()

    app = create_app()
    print("[MiniVecDB] Web UI ready at http://localhost:5000")
    print(f"[MiniVecDB] Shared DB: {get_shared_db_path()}")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
