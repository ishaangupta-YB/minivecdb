"""
+===============================================================+
|  MiniVecDB -- VectorStore (Core Engine)                        |
|  File: minivecdb/core/vector_store.py                          |
|                                                                |
|  Status: Day 11 complete                                         |
|                                                                |
|  This class integrates the full MiniVecDB feature set:         |
|    1) CRUD for records (insert, get, update, delete)           |
|    2) Vector search engine (cosine/euclidean/dot)              |
|    3) Metadata pre-filtering + collection-aware search          |
|    4) Collection management and bulk operations                 |
|    5) Robust persistence and restart recovery                   |
|       - vectors.npy (NumPy matrix)                              |
|       - id_mapping.json (row -> record_id bridge)              |
|       - SQLite (records, metadata, collections)                |
|    6) Context manager support (`with VectorStore(...) as db`)  |
+===============================================================+

HOW THE PIECES FIT TOGETHER:

    When you call store.insert("Hello world"):

    1. EmbeddingEngine converts text into a (384,) float32 vector.
    2. DatabaseManager stores text + metadata in SQLite.
    3. NumPy appends the vector as a new row in _vectors.
    4. _id_list stores the matching record ID for that row.
    5. save() persists vectors.npy + id_mapping.json.

If persistence files are missing/corrupt, _load_vectors() triggers
_rebuild_vectors() to re-embed from SQLite and restore consistency.
"""

import os
import json
import time
import logging
import shutil
import numpy as np
from typing import Optional, Dict, Any, List, Set

# ---------------------------------------------------------------
# Import project modules.
# We add the project root to sys.path so Python can find
# ARCHITECTURE.py (at the root) and sibling packages like storage/.
# ---------------------------------------------------------------
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ARCHITECTURE import (
    VectorRecord, SearchResult, CollectionInfo, DatabaseStats, generate_id,
)
from core.embeddings import create_embedding_engine
from core.runtime_paths import (
    get_model_cache_path,
    get_project_root,
    read_active_run_path,
    resolve_storage_path,
    create_new_run_path,
)
from core.distance_metrics import get_metric
from storage.database import DatabaseManager


logger = logging.getLogger(__name__)


class VectorStore:
    """
    The main class of MiniVecDB -- a mini vector database.

    It combines three storage layers:
      - SQLite  for structured data  (records, metadata, collections)
      - NumPy   for vector data      (N x 384 float32 matrix)
      - JSON    for the bridge       (row index -> record ID mapping)

    Attributes:
        storage_path:     Directory where all data files live.
        collection_name:  Default collection for insert/search.
        dimension:        Vector dimension (384 for all-MiniLM-L6-v2).
        db:               DatabaseManager instance (SQLite wrapper).
        embedding_engine: Converts text -> vectors.

    Example:
        store = VectorStore("./my_data")
        rid = store.insert("The cat sat on the mat")
        record = store.get(rid)
        print(record.text)   # "The cat sat on the mat"
        print(record.vector) # array([0.12, -0.03, ...])
    """

    # Default configuration constants
    DEFAULT_COLLECTION = "default"
    DEFAULT_DIMENSION = 384
    DEFAULT_RUN_PREFIX = "demo"

    def __init__(
        self,
        storage_path: Optional[str] = None,
        collection_name: str = "default",
        dimension: int = 384,
        embedding_model: Optional[str] = None,
        new_run: bool = False,
        run_prefix: str = DEFAULT_RUN_PREFIX,
        model_cache_path: Optional[str] = None,
    ) -> None:
        """
        Initialise the VectorStore.

                This constructor does quite a lot:
                    1. Resolves the storage directory (managed db_run path by default).
                    2. Creates the storage directory if it doesn't exist.
                    3. Opens a SQLite connection via DatabaseManager.
                    4. Ensures the default collection exists in SQLite.
                    5. Initialises the embedding engine (with fallback).
                    6. Loads any previously saved vectors from disk.

        Args:
            storage_path:    Directory to store all data files.
                             If None, uses managed project storage under
                             ./db_run/<run_name> and reuses active run.
                             Example explicit path: "./my_data"
            collection_name: Default collection name for inserts.
                             Default: "default"
            dimension:       Dimensionality of vectors.
                             Must match the embedding model (384).
            embedding_model: Optional model name override.
                             Default None uses "all-MiniLM-L6-v2".
            new_run:         If True and storage_path is None, force a
                             fresh unique run directory in ./db_run/.
            run_prefix:      Prefix for managed run directories.
                             Default: "demo"
            model_cache_path:Optional embedding model cache path.
                             If None, uses ./db_run/model_cache/huggingface.

        Raises:
            ValueError: If configuration values are invalid.
            ValueError: If persisted SQLite/vector state is inconsistent.
        """
        if storage_path is not None:
            self._require_non_empty_string(storage_path, "storage_path")
        self._require_non_empty_string(collection_name, "collection_name")
        self._require_non_empty_string(run_prefix, "run_prefix")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer.")

        # Resolve storage path.
        # Default behavior uses managed project storage under ./db_run and
        # reuses the active run directory across CLI commands.
        if storage_path is None and not new_run:
            active_run = read_active_run_path()
            if active_run is None:
                migrated = self._maybe_migrate_legacy_storage(run_prefix=run_prefix)
                if migrated is not None:
                    resolved_storage_path = migrated
                else:
                    resolved_storage_path = resolve_storage_path(
                        storage_path=None,
                        create_new_run=False,
                        run_prefix=run_prefix,
                    )
            else:
                resolved_storage_path = active_run
        else:
            resolved_storage_path = resolve_storage_path(
                storage_path=storage_path,
                create_new_run=(new_run and storage_path is None),
                run_prefix=run_prefix,
            )

        if model_cache_path is not None:
            self._require_non_empty_string(model_cache_path, "model_cache_path")
            resolved_cache_path = os.path.abspath(
                os.path.expanduser(model_cache_path)
            )
            os.makedirs(resolved_cache_path, exist_ok=True)
        else:
            resolved_cache_path = get_model_cache_path()

        # Store configuration for later use.
        self.storage_path: str = os.path.abspath(resolved_storage_path)
        self.collection_name: str = collection_name
        self.dimension: int = dimension
        self.run_prefix: str = run_prefix
        self.model_cache_path: str = resolved_cache_path

        # --- Step 1: Create storage directory ---
        # os.makedirs with exist_ok=True is safe to call repeatedly.
        # It creates the directory and any parent directories needed,
        # and does nothing if the directory already exists.
        os.makedirs(self.storage_path, exist_ok=True)

        # --- Step 2: Initialise SQLite via DatabaseManager ---
        # The database file lives inside our storage directory.
        db_path = os.path.join(self.storage_path, "minivecdb.db")
        self.db: DatabaseManager = DatabaseManager(db_path)

        # --- Step 3: Ensure default collection exists ---
        # The SCHEMA_SQL already creates a "default" collection, but
        # if the user specified a custom collection_name, we need to
        # create that one too.
        if not self.db.collection_exists(collection_name):
            self.db.create_collection(collection_name, dimension)

        # --- Step 4: Initialise embedding engine ---
        # create_embedding_engine() tries to load the real
        # SentenceTransformer model first.  If sentence-transformers
        # isn't installed, fallback=True makes it return the simpler
        # bag-of-words SimpleEmbeddingEngine instead.
        self.embedding_engine = create_embedding_engine(
            model_name=embedding_model,
            fallback=True,
            cache_folder=self.model_cache_path,
        )

        # --- Step 5: Initialise in-memory vector storage ---
        # _vectors is a 2D NumPy array of shape (N, 384) where N is
        # the number of stored records.  We start with an empty array
        # of shape (0, 384) -- zero rows, 384 columns.
        #
        # _id_list is a Python list that maps row indices to record IDs.
        # _id_list[i] is the record ID stored at _vectors[i].
        self._vectors: np.ndarray = np.empty(
            (0, dimension), dtype=np.float32
        )
        self._id_list: List[str] = []
        self._id_to_index: Dict[str, int] = {}

        # --- Step 6: Load existing vectors from disk ---
        # If we previously saved vectors.npy and id_mapping.json,
        # this restores them into _vectors and _id_list.
        self._load_vectors()
        self._rebuild_id_index()
        self._validate_internal_state()

    @staticmethod
    def _require_non_empty_string(value: str, field_name: str) -> None:
        """Validate that a field is a non-empty string."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        """Validate and normalize metadata into string key-value pairs."""
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary of key-value pairs.")

        normalized: Dict[str, str] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata keys must be non-empty strings.")
            normalized[key] = str(value)
        return normalized

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        """Validate that top_k is a positive integer value."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

    @staticmethod
    def _maybe_migrate_legacy_storage(run_prefix: str) -> Optional[str]:
        """Migrate old default storage folders into managed db_run storage.

        This preserves existing user data when upgrading from earlier
        defaults that used ./minivecdb_data or ./vectorstore_data.
        """
        project_root = get_project_root()
        legacy_dirs = (
            os.path.join(project_root, "minivecdb_data"),
            os.path.join(project_root, "vectorstore_data"),
        )
        required_files = ("minivecdb.db", "vectors.npy", "id_mapping.json")
        optional_files = ("vectors.npy.tmp", "id_mapping.json.tmp")

        for legacy_dir in legacy_dirs:
            if not os.path.isdir(legacy_dir):
                continue

            has_runtime_files = any(
                os.path.exists(os.path.join(legacy_dir, filename))
                for filename in required_files
            )
            if not has_runtime_files:
                continue

            target_dir = create_new_run_path(prefix=run_prefix)
            try:
                for filename in required_files + optional_files:
                    source_path = os.path.join(legacy_dir, filename)
                    if os.path.isfile(source_path):
                        shutil.copy2(
                            source_path,
                            os.path.join(target_dir, filename),
                        )
            except Exception as exc:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise ValueError(
                    "Failed to migrate legacy storage from "
                    f"{legacy_dir!r} into managed db_run storage. "
                    "Pass an explicit storage_path (or --db-path) to continue."
                ) from exc

            logger.info(
                "Migrated legacy storage from %s to %s",
                legacy_dir,
                target_dir,
            )
            return target_dir

        return None

    def _rebuild_id_index(self) -> None:
        """Rebuild O(1) ID to row-index mapping from the ID list."""
        self._id_to_index = {
            record_id: idx for idx, record_id in enumerate(self._id_list)
        }

    def _validate_internal_state(self) -> None:
        """
        Validate consistency between SQLite records and vector storage.

        After _load_vectors() has already handled major inconsistencies
        (triggering rebuild if needed), this method performs a final
        sanity check on the in-memory state.  If anything is still
        wrong at this point, it triggers one more rebuild as a last
        resort, then raises if rebuild didn't fix it.
        """
        problems = []

        if self._vectors.ndim != 2:
            problems.append("vectors must be a 2D array")
        elif self._vectors.shape[1] != self.dimension:
            problems.append(
                f"dimension mismatch: got {self._vectors.shape[1]}, "
                f"expected {self.dimension}"
            )

        if self._vectors.shape[0] != len(self._id_list):
            problems.append(
                f"row count {self._vectors.shape[0]} != "
                f"id_list length {len(self._id_list)}"
            )

        if len(set(self._id_list)) != len(self._id_list):
            problems.append("duplicate IDs in id_list")

        db_count = self.db.count_records()
        if db_count != len(self._id_list):
            problems.append(
                f"SQLite has {db_count} records but id_list has "
                f"{len(self._id_list)}"
            )

        if problems:
            logger.warning(
                "Post-load validation failed (%s). "
                "Attempting rebuild.",
                "; ".join(problems),
            )
            self._rebuild_vectors()
            self._rebuild_id_index()

    @staticmethod
    def _find_duplicate_ids(ids: List[str]) -> Set[str]:
        """Return a set of duplicated IDs found in the given list."""
        seen: Set[str] = set()
        duplicates: Set[str] = set()
        for record_id in ids:
            if record_id in seen:
                duplicates.add(record_id)
            seen.add(record_id)
        return duplicates

    # ===============================================================
    # INSERT -- Add a single record
    # ===============================================================

    def insert(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        collection: Optional[str] = None,
    ) -> str:
        """
        Insert a single text record into the vector store.

        This is the most commonly used method.  It:
          1. Generates a unique ID (if not provided).
          2. Checks for duplicate IDs (raises ValueError).
          3. Embeds the text into a 384-dim vector.
          4. Stores the record in SQLite (records + metadata tables).
          5. Appends the vector to the in-memory NumPy matrix.
          6. Saves everything to disk (vectors.npy + id_mapping.json).

        Args:
            text:       The text to store and embed.
                        Example: "The cat sat on the mat"
            metadata:   Optional key-value tags for filtering.
                        Example: {"category": "animals", "lang": "en"}
            id:         Optional custom ID string.
                        If None, auto-generates like "vec_a1b2c3d4".
            collection: Which collection to insert into.
                        Defaults to self.collection_name ("default").

        Returns:
            The ID of the newly inserted record (str).

        Raises:
            ValueError: If a record with the given ID already exists.
                        This prevents accidental overwrites.

        Example:
            rid = store.insert(
                "Python is great for data science",
                metadata={"topic": "programming", "year": "2024"},
            )
            print(rid)  # "vec_f3e8a1c2"
        """
        self._require_non_empty_string(text, "text")

        # --- Use defaults for optional args ---
        collection = collection or self.collection_name
        self._require_non_empty_string(collection, "collection")

        if not self.db.collection_exists(collection):
            raise ValueError(
                f"Collection '{collection}' does not exist. "
                "Create it before inserting records into it."
            )

        metadata = self._normalize_metadata(metadata or {})

        # --- Generate a unique ID if none was given ---
        # generate_id() creates IDs like "vec_a1b2c3d4" using uuid4.
        if id is None:
            id = generate_id()
        self._require_non_empty_string(id, "id")

        # --- Guard against duplicate IDs ---
        # We check the SQLite database, not the in-memory list, because
        # SQLite is the authoritative source of record existence.
        if self.db.record_exists(id):
            raise ValueError(
                f"Record with ID '{id}' already exists. "
                f"Each record must have a unique ID."
            )

        # --- Embed the text ---
        # This converts the text string into a 384-dimensional float32
        # NumPy array.  The embedding captures the MEANING of the text,
        # so similar texts produce similar vectors.
        vector = self.embedding_engine.encode(text)
        vector = np.asarray(vector, dtype=np.float32)

        if vector.ndim != 1 or vector.shape[0] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected ({self.dimension},), got {vector.shape}."
            )

        created_at = time.time()
        try:
            # --- Store structured data in SQLite atomically ---
            with self.db.transaction():
                self.db.insert_record(
                    id,
                    text,
                    collection,
                    created_at,
                    auto_commit=False,
                )
                # Each metadata key-value pair goes into the "metadata"
                # table as a separate row (EAV pattern).
                for key, value in metadata.items():
                    self.db.insert_metadata(
                        id,
                        key,
                        value,
                        auto_commit=False,
                    )
        except Exception as exc:
            raise ValueError(f"Failed to insert record '{id}': {exc}") from exc

        previous_vectors = self._vectors
        previous_id_list = list(self._id_list)

        # --- Append vector to in-memory NumPy matrix ---
        # reshape(1, -1) converts the 1D vector (384,) into a 2D
        # row vector (1, 384) so it can be vertically stacked.
        vector_2d = vector.reshape(1, -1)

        if self._vectors.size == 0:
            # First record: just use this vector as the entire matrix.
            self._vectors = vector_2d
        else:
            # Subsequent records: stack on top of existing matrix.
            # np.vstack joins arrays vertically:
            #   (N, 384) + (1, 384)  -->  (N+1, 384)
            self._vectors = np.vstack([self._vectors, vector_2d])

        # --- Update the ID mapping ---
        # After vstack, the new vector is at the LAST row.
        # So _id_list.append(id) maps that last row to this record.
        self._id_list.append(id)
        self._rebuild_id_index()

        try:
            # --- Persist to disk ---
            # Auto-save after every insert ensures no data is lost if
            # the program crashes.
            self.save()
        except Exception as exc:
            # Revert in-memory state first.
            self._vectors = previous_vectors
            self._id_list = previous_id_list
            self._rebuild_id_index()

            # Compensating action: remove the SQLite row that was inserted.
            self.db.delete_record(id)
            raise RuntimeError(
                "Insert failed while saving vector files. "
                "SQLite insert was rolled back via compensating delete."
            ) from exc

        return id

    # ===============================================================
    # INSERT BATCH -- Add multiple records at once (fast)
    # ===============================================================

    def insert_batch(
        self,
        texts: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection: Optional[str] = None,
    ) -> List[str]:
        """
        Insert multiple records at once using batch embedding.

        This is MUCH faster than calling insert() in a loop because:
          - encode_batch() processes all texts through the neural
            network in one pass (GPU parallelism, less overhead).
          - We only call save() once at the end instead of N times.

        For 1000 records, insert_batch is typically 10-50x faster
        than 1000 individual insert() calls.

        Args:
            texts:         List of text strings to insert.
            metadata_list: Optional list of metadata dicts, one per text.
                           If None, all records get empty metadata {}.
            ids:           Optional list of custom IDs.
                           If None, auto-generates all IDs.
            collection:    Collection to insert into.

        Returns:
            List of IDs for the inserted records (same order as texts).

        Raises:
            ValueError: If any ID already exists in the database.
            ValueError: If list lengths don't match.

        Example:
            ids = store.insert_batch(
                texts=["Hello world", "Goodbye world"],
                metadata_list=[{"lang": "en"}, {"lang": "en"}],
            )
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list of strings.")

        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string.")

        collection = collection or self.collection_name
        self._require_non_empty_string(collection, "collection")
        if not self.db.collection_exists(collection):
            raise ValueError(
                f"Collection '{collection}' does not exist. "
                "Create it before inserting records into it."
            )

        n = len(texts)

        # --- Fill in defaults ---
        if metadata_list is None:
            metadata_list = [{} for _ in range(n)]
        if ids is None:
            ids = [generate_id() for _ in range(n)]

        # --- Validate list lengths match ---
        if len(metadata_list) != n:
            raise ValueError(
                f"metadata_list length ({len(metadata_list)}) "
                f"doesn't match texts length ({n}). "
                f"Provide one metadata dict per text."
            )
        if len(ids) != n:
            raise ValueError(
                f"ids length ({len(ids)}) doesn't match "
                f"texts length ({n}). Provide one ID per text."
            )

        normalized_metadata_list = [
            self._normalize_metadata(item) for item in metadata_list
        ]

        for i, record_id in enumerate(ids):
            self._require_non_empty_string(record_id, f"ids[{i}]")

        duplicate_ids = self._find_duplicate_ids(ids)
        if duplicate_ids:
            raise ValueError(
                "Duplicate IDs were provided in the same batch: "
                f"{sorted(duplicate_ids)}"
            )

        # --- Check for duplicate IDs before doing any work ---
        # We check ALL IDs first so we don't leave the database in a
        # half-inserted state if a duplicate is found partway through.
        for record_id in ids:
            if self.db.record_exists(record_id):
                raise ValueError(
                    f"Record with ID '{record_id}' already exists."
                )

        # --- Batch embed all texts at once ---
        # This is the BIG speed win.  Instead of N separate calls to
        # the neural network, we send all texts at once.
        vectors = self.embedding_engine.encode_batch(texts)
        vectors = np.asarray(vectors, dtype=np.float32)

        if vectors.ndim != 2 or vectors.shape != (n, self.dimension):
            raise ValueError(
                "Embedding dimension mismatch for batch insert: "
                f"expected {(n, self.dimension)}, got {vectors.shape}."
            )

        created_at = time.time()
        try:
            # --- Insert each record into SQLite atomically ---
            with self.db.transaction():
                for i in range(n):
                    self.db.insert_record(
                        ids[i],
                        texts[i],
                        collection,
                        created_at,
                        auto_commit=False,
                    )
                    for key, value in normalized_metadata_list[i].items():
                        self.db.insert_metadata(
                            ids[i],
                            key,
                            value,
                            auto_commit=False,
                        )
        except Exception as exc:
            raise ValueError(f"Failed to insert batch records: {exc}") from exc

        previous_vectors = self._vectors
        previous_id_list = list(self._id_list)

        # --- Append all vectors to in-memory matrix at once ---
        new_vectors = vectors
        if self._vectors.size == 0:
            self._vectors = new_vectors
        else:
            self._vectors = np.vstack([self._vectors, new_vectors])

        # --- Extend the ID mapping ---
        self._id_list.extend(ids)
        self._rebuild_id_index()

        try:
            # --- Persist to disk (once, not N times) ---
            self.save()
        except Exception as exc:
            # Revert in-memory state first.
            self._vectors = previous_vectors
            self._id_list = previous_id_list
            self._rebuild_id_index()

            # Compensating action: delete inserted SQLite rows.
            cleanup_errors: List[str] = []
            for record_id in ids:
                try:
                    self.db.delete_record(record_id)
                except Exception as cleanup_exc:
                    cleanup_errors.append(f"{record_id}: {cleanup_exc}")

            if cleanup_errors:
                raise RuntimeError(
                    "Batch insert failed while saving vector files, and "
                    "cleanup could not delete some SQLite records: "
                    f"{cleanup_errors}"
                ) from exc

            raise RuntimeError(
                "Batch insert failed while saving vector files. "
                "SQLite rows were removed via compensating delete."
            ) from exc

        return ids

    # ===============================================================
    # GET -- Retrieve a single record by ID
    # ===============================================================

    def get(self, id: str) -> Optional[VectorRecord]:
        """
        Retrieve a complete VectorRecord by its ID.

        This method reassembles a full VectorRecord by pulling data
        from three sources:
          1. SQLite records table  -> id, text, collection, created_at
          2. SQLite metadata table -> {key: value, ...}
          3. NumPy _vectors array  -> 384-dim float32 vector

        Args:
            id: The record ID to look up (e.g. "vec_a1b2c3d4").

        Returns:
            A VectorRecord dataclass with all fields populated,
            or None if no record with that ID exists.

        Example:
            record = store.get("vec_a1b2c3d4")
            if record:
                print(record.text)       # "The cat sat on the mat"
                print(record.metadata)   # {"category": "animals"}
                print(record.vector[:5]) # [0.12, -0.03, 0.45, ...]
        """
        self._require_non_empty_string(id, "id")

        # --- Step 1: Fetch the record from SQLite ---
        row = self.db.get_record(id)
        if row is None:
            return None

        # --- Step 2: Fetch metadata from SQLite ---
        metadata = self.db.get_metadata(id)

        # --- Step 3: Find the vector in the NumPy matrix ---
        # We use _id_list.index(id) to find which row in _vectors
        # belongs to this record.  For example, if _id_list[42] == id,
        # then _vectors[42] is the vector we want.
        idx = self._id_to_index.get(id)
        if idx is None or idx >= self._vectors.shape[0]:
            raise RuntimeError(
                "Data integrity error: record exists in SQLite but is missing "
                "from vector storage mapping."
            )
        vector = self._vectors[idx]

        # --- Step 4: Build and return a VectorRecord ---
        # VectorRecord.from_db_row() is a classmethod defined in
        # ARCHITECTURE.py that constructs the dataclass from a
        # database row tuple, a vector, and a metadata dict.
        return VectorRecord.from_db_row(row, vector, metadata)

    # ===============================================================
    # SAVE / LOAD -- Disk persistence for vectors
    # ===============================================================
    #
    # SQLite auto-persists (commit after every write), but our NumPy
    # vectors live in memory.  save() and _load_vectors() handle
    # writing/reading the vector data to/from disk files.
    #
    # Files:
    #   vectors.npy      -- NumPy binary format, shape (N, 384)
    #   id_mapping.json  -- JSON list mapping row index -> record ID
    #
    # ===============================================================

    def save(self) -> str:
        """
        Persist the in-memory vectors and ID mapping to disk.

        Writes two files into self.storage_path:
          - vectors.npy:      The full (N, 384) float32 NumPy matrix.
          - id_mapping.json:  A JSON list like ["vec_001", "vec_002", ...].

        Handles the edge case of an empty database (zero records) by
        saving an empty (0, 384) array and an empty JSON list [].

        SQLite auto-persists on commit, so no action is needed for
        the structured data — this method only saves the vector files.

        Returns:
            The storage_path where files were saved.

        Why two files?
            NumPy's .npy format is compact and fast for loading arrays,
            but it can't store string IDs.  So we use a separate JSON
            file for the ID list.  Together they form the "bridge"
            between the NumPy row indices and SQLite record IDs.
        """
        vectors_path = os.path.join(self.storage_path, "vectors.npy")
        mapping_path = os.path.join(self.storage_path, "id_mapping.json")
        vectors_tmp_path = os.path.join(self.storage_path, "vectors.npy.tmp")
        mapping_tmp_path = os.path.join(self.storage_path, "id_mapping.json.tmp")

        try:
            # --- Handle empty database ---
            # np.save works fine with a (0, 384) array, and json.dump
            # works fine with an empty list [].  No special case needed.
            with open(vectors_tmp_path, "wb") as vector_file:
                np.save(vector_file, self._vectors)
            with open(mapping_tmp_path, "w", encoding="utf-8") as mapping_file:
                json.dump(self._id_list, mapping_file)

            # Atomic replace: rename temp -> final.  If we crash between
            # the two renames, at worst ONE file is stale, and _load_vectors
            # will detect the inconsistency and trigger a rebuild.
            os.replace(vectors_tmp_path, vectors_path)
            os.replace(mapping_tmp_path, mapping_path)
        except Exception:
            # Best-effort cleanup of temp files on failure.
            if os.path.exists(vectors_tmp_path):
                os.remove(vectors_tmp_path)
            if os.path.exists(mapping_tmp_path):
                os.remove(mapping_tmp_path)
            raise

        return self.storage_path

    def _load_vectors(self) -> None:
        """
        Load previously saved vectors and ID mapping from disk.

        Called during __init__() to restore state from a prior session.

        Consistency checks (in order):
          1. Both files must exist.  If either is missing but SQLite has
             records, trigger a rebuild (the .npy was lost or corrupted).
          2. id_mapping.json must be a valid JSON list of strings.
          3. Row count of vectors.npy must equal len(id_mapping.json).
          4. Every ID in id_mapping.json must exist in SQLite.

        If any check fails, log a warning and call _rebuild_vectors()
        to re-create the vector files from SQLite source data.
        """
        vectors_path = os.path.join(self.storage_path, "vectors.npy")
        mapping_path = os.path.join(self.storage_path, "id_mapping.json")

        db_count = self.db.count_records()

        # --- Case 1: No saved files ---
        if not os.path.exists(vectors_path) or not os.path.exists(mapping_path):
            if db_count > 0:
                # SQLite has records but vector files are missing.
                # This happens if the .npy was deleted or corrupted.
                logger.warning(
                    "Vector files missing but SQLite has %d records. "
                    "Rebuilding vectors from SQLite.",
                    db_count,
                )
                self._rebuild_vectors()
            # else: fresh database, nothing to load — keep empty defaults.
            return

        # --- Case 2: Files exist — load them ---
        try:
            self._vectors = np.load(
                vectors_path, allow_pickle=False
            ).astype(np.float32)

            with open(mapping_path, "r", encoding="utf-8") as f:
                self._id_list = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load vector files (%s). Rebuilding from SQLite.",
                exc,
            )
            self._rebuild_vectors()
            return

        # --- Validate id_mapping.json format ---
        if not isinstance(self._id_list, list):
            logger.warning(
                "id_mapping.json is not a list. Rebuilding from SQLite."
            )
            self._rebuild_vectors()
            return

        for i, record_id in enumerate(self._id_list):
            if not isinstance(record_id, str) or not record_id.strip():
                logger.warning(
                    "id_mapping.json has invalid ID at index %d. "
                    "Rebuilding from SQLite.",
                    i,
                )
                self._rebuild_vectors()
                return

        # --- Check row count consistency ---
        if self._vectors.shape[0] != len(self._id_list):
            logger.warning(
                "vectors.npy has %d rows but id_mapping.json has %d entries. "
                "Rebuilding from SQLite.",
                self._vectors.shape[0],
                len(self._id_list),
            )
            self._rebuild_vectors()
            return

        # --- Check all IDs exist in SQLite ---
        db_ids = set(self.db.get_all_record_ids())
        mapping_ids = set(self._id_list)

        if mapping_ids != db_ids:
            logger.warning(
                "ID mismatch: %d IDs in mapping, %d IDs in SQLite. "
                "Rebuilding from SQLite.",
                len(mapping_ids),
                len(db_ids),
            )
            self._rebuild_vectors()
            return

    def _rebuild_vectors(self) -> None:
        """
        Emergency rebuild: re-create the vector matrix from SQLite.

        This is the "nuclear option" for when vector files (.npy, .json)
        are corrupted, deleted, or out of sync with SQLite.

        Algorithm:
          1. Get all record IDs and texts from SQLite (the authoritative
             source of truth for what records exist).
          2. Re-embed every text through the embedding engine.
          3. Build a new _vectors matrix and _id_list from scratch.
          4. Save the rebuilt data to disk.

        If the database is empty, resets to empty arrays.

        This can be slow for large databases because it re-embeds every
        text, but it guarantees correctness.
        """
        records = self.db.get_all_records_with_text()

        if not records:
            # Empty database — reset to empty arrays.
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)
            self._id_list = []
            self._rebuild_id_index()
            self.save()
            return

        # --- Re-embed all texts ---
        ids = [row[0] for row in records]
        texts = [row[1] for row in records]

        vectors = self.embedding_engine.encode_batch(texts)
        vectors = np.asarray(vectors, dtype=np.float32)

        # --- Replace in-memory state ---
        self._vectors = vectors
        self._id_list = ids
        self._rebuild_id_index()

        # --- Save the rebuilt data to disk ---
        self.save()

        logger.info(
            "Rebuilt %d vectors from SQLite successfully.", len(ids)
        )

    # ===============================================================
    # COUNT -- Delegate to DatabaseManager
    # ===============================================================

    def count(self, collection: Optional[str] = None) -> int:
        """
        Count records in the database.

        This delegates to DatabaseManager.count_records(), which
        runs a SELECT COUNT(*) query on SQLite.

        Args:
            collection: If given, count only records in this collection.
                        If None, count all records across all collections.

        Returns:
            Integer count of matching records.

        Example:
            total = store.count()               # all records
            sci   = store.count("science")      # just "science" collection
        """
        return self.db.count_records(collection)

    def __len__(self) -> int:
        """
        Support len(store) syntax.

        Returns the total number of records, same as count().
        This lets VectorStore work anywhere Python expects a "sized"
        object, e.g. len(store), bool(store), etc.
        """
        return self.count()

    # ===============================================================
    # SEARCH -- The core feature of a vector database
    # ===============================================================
    #
    # This is what makes MiniVecDB a *vector* database rather than
    # just a regular database.  The search engine:
    #   1. Converts the user's query text into a vector
    #   2. Compares that vector against every stored vector
    #   3. Returns the most similar records, ranked by score
    #
    # The comparison uses one of three metrics (cosine, euclidean,
    # dot product) implemented in distance_metrics.py.  All metrics
    # use NumPy batch operations for speed.
    #
    # Optional filters let users narrow the search to records that
    # match specific metadata tags BEFORE computing similarity.
    # This is called "pre-filtering" and reduces the number of
    # expensive vector comparisons needed.
    #
    # ===============================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        metric: str = "cosine",
        filters: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search the database for records most similar to a query text.

        This is the main search method.  It:
          1. Validates the database is not empty.
          2. Embeds the query text into a 384-dim vector.
          3. Determines candidate vectors (all, or filtered subset).
          4. Computes similarity scores using the chosen metric.
          5. Sorts by relevance and takes the top_k results.
          6. Builds SearchResult objects with rank, score, and record.

        Args:
            query:      The text to search for.
                        Example: "machine learning algorithms"
            top_k:      How many results to return (default 5).
                        The actual number returned may be less if
                        the database has fewer matching records.
            metric:     Which similarity metric to use.
                        "cosine"    — direction-based (default, best for text)
                        "euclidean" — distance-based (lower = better)
                        "dot"       — fastest (best with normalised vectors)
            filters:    Optional metadata filters (AND logic).
                        Supports exact match, list (OR), and operators
                        ($gt, $lt, $gte, $lte, $ne).
                        Example: {"category": "science", "year": {"$gt": "2020"}}
                        Only records matching ALL filters are searched.
            collection: Optional collection name to restrict search to.

        Returns:
            List of SearchResult objects, sorted by relevance.
            Each SearchResult has: .record, .score, .rank, .metric

        Raises:
            ValueError: If the database is empty (nothing to search).
            ValueError: If the metric name is not recognised.

        Example:
            results = store.search("machine learning", top_k=3)
            for r in results:
                print(f"#{r.rank} [{r.score:.4f}] {r.record.text}")
        """
        self._require_non_empty_string(query, "query")
        self._validate_top_k(top_k)
        self._require_non_empty_string(metric, "metric")
        metric_name = metric.strip().lower()

        if self._vectors.shape[0] == 0:
            raise ValueError(
                "Cannot search an empty database. "
                "Insert some records first with store.insert()."
            )

        # --- Step 1: Look up the metric ---
        # get_metric() returns a dict with 'batch', 'higher_is_better', etc.
        # It raises ValueError if the metric name is unknown.
        metric_info = get_metric(metric_name)
        batch_fn = metric_info["batch"]
        higher_is_better = metric_info["higher_is_better"]

        # --- Step 2: Embed the query text ---
        # The embedding engine converts the query into a 384-dim vector
        # that can be compared against all stored vectors.
        query_vector = self.embedding_engine.encode(query)
        query_vector = np.asarray(query_vector, dtype=np.float32)

        # --- Step 3: Determine candidate vectors ---
        # If filters or collection are specified, we only search a
        # subset of vectors.  This is "pre-filtering" — it reduces
        # the number of similarity computations needed.
        if filters is not None:
            # SQL metadata filter → get matching row indices
            indices = self._get_filtered_indices(filters, collection)
            if len(indices) == 0:
                return []  # No records match the filters
            candidate_vectors = self._vectors[indices]
        elif collection is not None:
            # No metadata filters, but restrict to one collection
            col_ids = self.db.get_record_ids_in_collection(collection)
            if not col_ids:
                return []
            indices = np.array(
                [self._id_to_index[rid] for rid in col_ids if rid in self._id_to_index],
                dtype=np.intp,
            )
            if len(indices) == 0:
                return []
            candidate_vectors = self._vectors[indices]
        else:
            # Search ALL vectors (no filtering)
            indices = np.arange(self._vectors.shape[0])
            candidate_vectors = self._vectors

        # --- Step 4: Compute similarity scores ---
        # The batch function computes the query vs every candidate
        # in one vectorised operation.  Much faster than a Python loop.
        scores = batch_fn(query_vector, candidate_vectors)

        # --- Step 5: Sort by relevance ---
        # For cosine and dot product, higher is better → sort descending.
        # For euclidean distance, lower is better → sort ascending.
        if higher_is_better:
            sorted_positions = np.argsort(scores)[::-1]  # descending
        else:
            sorted_positions = np.argsort(scores)  # ascending

        # --- Step 6: Take top_k and build SearchResult objects ---
        # Clamp top_k to the number of candidates available.
        top_k = min(top_k, len(sorted_positions))
        results: List[SearchResult] = []

        for rank_idx in range(top_k):
            # sorted_positions[rank_idx] is the position within the
            # candidate array.  indices[that position] maps back to
            # the original row in _vectors / _id_list.
            candidate_pos = sorted_positions[rank_idx]
            original_row = int(indices[candidate_pos])
            record_id = self._id_list[original_row]

            # Fetch the full record from SQLite + NumPy
            record = self.get(record_id)
            if record is None:
                continue  # Shouldn't happen, but be defensive

            results.append(SearchResult(
                record=record,
                score=float(scores[candidate_pos]),
                rank=rank_idx + 1,  # 1-based rank
                metric=metric_name,
            ))

        return results

    # ===============================================================
    # FILTERED SEARCH HELPER
    # ===============================================================

    def _get_filtered_indices(
        self,
        filters: Dict[str, Any],
        collection: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get NumPy row indices for records matching metadata filters.

        This is the bridge between SQL-based metadata filtering and
        NumPy-based vector search.  It:
          1. Queries SQLite for record IDs matching the filters.
          2. Optionally intersects with a collection's IDs.
          3. Converts those IDs to NumPy matrix row indices.

        Args:
            filters:    Metadata filters (AND logic). Supports exact
                        match, list (OR), and operators ($gt, $lt, etc.).
            collection: Optional collection to further restrict to.

        Returns:
            NumPy array of integer indices into self._vectors.
            Empty array if nothing matches.
        """
        # Step 1: Get IDs matching metadata filters via SQL
        matching_ids = set(self.db.filter_by_metadata(filters))

        # Step 2: If a collection is specified, intersect with it
        if collection is not None:
            col_ids = set(self.db.get_record_ids_in_collection(collection))
            matching_ids = matching_ids & col_ids

        if not matching_ids:
            return np.array([], dtype=np.intp)

        # Step 3: Convert record IDs to row indices in _vectors
        # _id_to_index maps each record ID to its row in the matrix.
        indices = [
            self._id_to_index[rid]
            for rid in matching_ids
            if rid in self._id_to_index
        ]

        return np.array(sorted(indices), dtype=np.intp)

    # ===============================================================
    # DELETE -- Remove a single record by ID
    # ===============================================================

    def delete(self, id: str) -> bool:
        """
        Delete a record from both SQLite and the NumPy vector matrix.

        This removes data from three places:
          1. SQLite records table (cascade deletes metadata too)
          2. The in-memory _vectors NumPy matrix (removes one row)
          3. The _id_list bridge mapping

        Then saves the updated vectors to disk.

        Args:
            id: The record ID to delete (e.g. "vec_a1b2c3d4").

        Returns:
            True  if the record was found and deleted,
            False if no record with that ID existed.

        Example:
            deleted = store.delete("vec_a1b2c3d4")
            if deleted:
                print("Record removed!")
        """
        self._require_non_empty_string(id, "id")

        # --- Check if the record exists ---
        if not self.db.record_exists(id):
            return False

        # --- Find the row index in the NumPy matrix ---
        # _id_to_index is our O(1) lookup from record ID to row number.
        idx = self._id_to_index.get(id)
        if idx is None:
            return False

        # --- Delete from SQLite ---
        # CASCADE ensures metadata rows are auto-deleted too.
        self.db.delete_record(id)

        # --- Remove from the NumPy matrix ---
        # np.delete(array, index, axis=0) removes one row and returns
        # a new array with shape (N-1, 384).
        self._vectors = np.delete(self._vectors, idx, axis=0)

        # --- Remove from the ID list ---
        self._id_list.pop(idx)

        # --- Rebuild the ID-to-index mapping ---
        # After removing a row, all indices after `idx` shift down by 1.
        # Rather than manually adjusting, we just rebuild the whole dict.
        self._rebuild_id_index()

        # --- Persist to disk ---
        self.save()

        return True

    # ===============================================================
    # UPDATE -- Modify an existing record's text and/or metadata
    # ===============================================================

    def update(
        self,
        id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing record's text and/or metadata.

        If text is provided, the record is re-embedded (new vector).
        If metadata is provided, it REPLACES all existing metadata
        (full replace, not merge — simpler and less error-prone).

        Args:
            id:       The record ID to update.
            text:     New text to replace the existing text.
                      If provided, the vector is also re-computed.
            metadata: New metadata dict to replace all existing metadata.
                      Pass {} to clear metadata, None to leave unchanged.

        Returns:
            True  if the record was updated,
            False if no record with that ID existed.

        Example:
            store.update("vec_abc", text="Updated text")
            store.update("vec_abc", metadata={"year": "2025"})
            store.update("vec_abc", text="New!", metadata={"k": "v"})
        """
        self._require_non_empty_string(id, "id")

        # --- Check if the record exists ---
        if not self.db.record_exists(id):
            return False

        # --- Update text (and re-embed) ---
        if text is not None:
            self._require_non_empty_string(text, "text")

            # Re-embed the new text into a fresh 384-dim vector.
            new_vector = self.embedding_engine.encode(text)
            new_vector = np.asarray(new_vector, dtype=np.float32)

            # Update the text in SQLite.
            self.db.update_record_text(id, text)

            # Replace the vector in the NumPy matrix.
            idx = self._id_to_index[id]
            self._vectors[idx] = new_vector

        # --- Update metadata (full replace) ---
        if metadata is not None:
            normalized = self._normalize_metadata(metadata)

            # Delete all existing metadata for this record, then
            # insert the new key-value pairs.  Full replace is simpler
            # than trying to diff/merge individual keys.
            self.db.delete_metadata(id)
            for key, value in normalized.items():
                self.db.insert_metadata(id, key, value)

        # --- Persist to disk ---
        self.save()

        return True

    # ===============================================================
    # COLLECTION MANAGEMENT
    # ===============================================================
    #
    # Collections group related records (like folders for files).
    # Every record belongs to exactly one collection.  The "default"
    # collection always exists and cannot be deleted.
    #
    # ===============================================================

    def create_collection(self, name: str, description: str = "") -> CollectionInfo:
        """
        Create a new collection for organising records.

        Args:
            name:        Unique collection name (e.g. "science_papers").
            description: Optional human-readable description.

        Returns:
            A CollectionInfo dataclass with the new collection's details.

        Raises:
            ValueError: If a collection with this name already exists.

        Example:
            info = store.create_collection("papers", "Research papers")
            print(info.name)  # "papers"
        """
        self._require_non_empty_string(name, "name")

        # --- Check for duplicates ---
        if self.db.collection_exists(name):
            raise ValueError(
                f"Collection '{name}' already exists. "
                "Choose a different name."
            )

        # --- Insert into SQLite ---
        self.db.create_collection(name, self.dimension, description)

        return CollectionInfo(
            name=name,
            dimension=self.dimension,
            count=0,
            created_at=time.time(),
            description=description,
        )

    def list_collections(self) -> List[CollectionInfo]:
        """
        List all collections with their record counts.

        Uses a LEFT JOIN so collections with zero records still appear.

        Returns:
            List of CollectionInfo dataclasses, one per collection.

        Example:
            for col in store.list_collections():
                print(f"{col.name}: {col.count} records")
        """
        rows = self.db.list_collections()

        # Each row from the SQL is:
        #   (name, dimension, description, created_at, record_count)
        return [
            CollectionInfo(
                name=row[0],
                dimension=row[1],
                count=row[4],
                created_at=row[3],
                description=row[2],
            )
            for row in rows
        ]

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection and ALL records in it.

        The "default" collection is protected and cannot be deleted —
        it serves as the fallback for records without an explicit
        collection.

        CASCADE delete in SQLite automatically removes:
          - All records in the collection
          - All metadata for those records

        We also remove the corresponding vectors from the NumPy
        matrix and ID list.

        Args:
            name: The collection name to delete.

        Returns:
            True  if the collection was deleted,
            False if it didn't exist.

        Raises:
            ValueError: If you try to delete the "default" collection.

        Example:
            store.delete_collection("old_papers")
        """
        self._require_non_empty_string(name, "name")

        # --- Protect the default collection ---
        if name == self.DEFAULT_COLLECTION:
            raise ValueError(
                "Cannot delete the 'default' collection. "
                "It is required by the system."
            )

        if not self.db.collection_exists(name):
            return False

        # --- Find which vectors belong to this collection ---
        # We need to remove them from the NumPy matrix BEFORE the
        # SQL cascade delete erases the records.
        col_ids = set(self.db.get_record_ids_in_collection(name))

        # --- Delete from SQLite (cascade removes records + metadata) ---
        self.db.delete_collection(name)

        # --- Remove vectors from NumPy matrix ---
        if col_ids:
            # Find the row indices for all records in this collection.
            indices_to_remove = sorted(
                self._id_to_index[rid]
                for rid in col_ids
                if rid in self._id_to_index
            )

            if indices_to_remove:
                # np.delete with a list of indices removes multiple rows.
                self._vectors = np.delete(
                    self._vectors, indices_to_remove, axis=0
                )

                # Remove IDs from the list (iterate in reverse so
                # indices don't shift as we remove).
                for idx in reversed(indices_to_remove):
                    self._id_list.pop(idx)

                self._rebuild_id_index()

        # --- Persist to disk ---
        self.save()

        return True

    # ===============================================================
    # LIST IDS -- Retrieve record IDs
    # ===============================================================

    def list_ids(
        self, collection: Optional[str] = None, limit: int = 100
    ) -> List[str]:
        """
        Get record IDs from the database, with optional filtering.

        Args:
            collection: If given, only return IDs from this collection.
            limit:      Maximum number of IDs to return (default 100).

        Returns:
            List of record ID strings.

        Example:
            all_ids = store.list_ids()
            sci_ids = store.list_ids(collection="science", limit=10)
        """
        return self.db.list_record_ids(collection=collection, limit=limit)

    # ===============================================================
    # CLEAR -- Delete all records (or all in a collection)
    # ===============================================================

    def clear(self, collection: Optional[str] = None) -> int:
        """
        Delete all records, or all records in a specific collection.

        This is a bulk-delete operation.  It clears:
          - SQLite records + metadata (via cascade)
          - The in-memory NumPy vector matrix
          - The ID mapping

        Args:
            collection: If given, clear only records in this collection.
                        If None, clear ALL records across all collections.

        Returns:
            Number of records deleted.

        Example:
            store.clear("scratch")   # clear one collection
            store.clear()            # clear everything
        """
        if collection is not None:
            self._require_non_empty_string(collection, "collection")

            # --- Find which IDs belong to this collection ---
            col_ids = set(self.db.get_record_ids_in_collection(collection))

            # --- Delete from SQLite ---
            deleted_count = self.db.delete_records_in_collection(collection)

            # --- Remove corresponding vectors ---
            if col_ids:
                indices_to_remove = sorted(
                    self._id_to_index[rid]
                    for rid in col_ids
                    if rid in self._id_to_index
                )

                if indices_to_remove:
                    self._vectors = np.delete(
                        self._vectors, indices_to_remove, axis=0
                    )
                    for idx in reversed(indices_to_remove):
                        self._id_list.pop(idx)
                    self._rebuild_id_index()
        else:
            # --- Clear ALL records across all collections ---
            deleted_count = self.db.delete_all_records()

            # Reset in-memory state to empty.
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)
            self._id_list = []
            self._rebuild_id_index()

        # --- Persist to disk ---
        self.save()

        return deleted_count

    # ===============================================================
    # STATS -- Database statistics
    # ===============================================================

    def stats(self) -> DatabaseStats:
        """
        Get overall database statistics.

        Returns a DatabaseStats dataclass with:
          - total_records:      Number of records across all collections
          - total_collections:  Number of collections
          - dimension:          Vector dimensionality (384)
          - memory_usage_bytes: Estimated memory for vectors (N * 384 * 4)
          - storage_path:       Directory where data files live
          - embedding_model:    Name of the embedding model in use
          - db_file:            Path to the SQLite database file

        The memory estimate uses N * dimension * 4 because each float32
        value occupies exactly 4 bytes.

        Example:
            s = store.stats()
            print(f"{s.total_records} records using {s.memory_usage_bytes} bytes")
        """
        total_records = self.db.count_records()
        total_collections = len(self.db.list_collections())

        # Memory estimate: each vector is dimension float32 values.
        # float32 = 4 bytes, so memory = N * 384 * 4.
        memory_bytes = total_records * self.dimension * 4

        return DatabaseStats(
            total_records=total_records,
            total_collections=total_collections,
            dimension=self.dimension,
            memory_usage_bytes=memory_bytes,
            storage_path=self.storage_path,
            embedding_model=getattr(
                self.embedding_engine, "model_name", "unknown"
            ),
            db_file=self.db.db_path,
        )

    # ===============================================================
    # CLOSE + CONTEXT MANAGER
    # ===============================================================
    #
    # close() saves vectors and shuts down the SQLite connection.
    # __enter__/__exit__ enable the `with VectorStore(...) as db:` pattern
    # so Python automatically calls close() when you leave the block.
    #
    # ===============================================================

    def close(self) -> None:
        """
        Save vectors to disk and close the SQLite connection.

        Always call this when you're done with the VectorStore.
        After close(), no further operations are possible.

        Example:
            store = VectorStore("./data")
            store.insert("Hello")
            store.close()  # saves and disconnects
        """
        self.save()
        self.db.close()

    def __enter__(self) -> "VectorStore":
        """
        Enter the context manager -- just return self.

        This enables the pattern:
            with VectorStore("./data") as store:
                store.insert("Hello")
            # close() is called automatically here
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the context manager -- calls close().

        This runs even if an exception occurred inside the `with` block,
        ensuring the database is always properly shut down.

        Args:
            exc_type: Exception class (or None if no error).
            exc_val:  Exception instance (or None).
            exc_tb:   Traceback object (or None).
        """
        self.close()
