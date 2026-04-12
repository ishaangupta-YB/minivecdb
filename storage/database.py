"""
+===============================================================+
|  MiniVecDB -- DatabaseManager (SQLite Wrapper)                 |
|  File: minivecdb/storage/database.py                           |
|                                                                |
|  This module is the ONLY place in the project that talks to    |
|  SQLite directly. Every other module (VectorStore, CLI, web)   |
|  goes through this class to read/write structured data.        |
|                                                                |
|  It manages three tables:                                      |
|    collections -- groups of related vectors (like folders)     |
|    records     -- the actual data entries (id, text, etc.)     |
|    metadata    -- key-value tags on records (EAV pattern)      |
|                                                                |
|  KEY CONCEPTS FOR LEARNING:                                    |
|    - Parameterised queries (?) prevent SQL injection           |
|    - Foreign keys with CASCADE auto-delete child rows          |
|    - EAV (Entity-Attribute-Value) allows flexible metadata     |
|    - All SQL lives in ARCHITECTURE.py, not here                |
+===============================================================+
"""

import sqlite3
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Tuple, Iterator

# ---------------------------------------------------------------
# Import the SQL schema and query templates from the central
# architecture file. ARCHITECTURE.py is at the project root
# (one directory above storage/), so we add the parent directory
# to Python's import search path.
# ---------------------------------------------------------------
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ARCHITECTURE import SCHEMA_SQL, SQL_QUERIES


class DatabaseManager:
    """
    Wraps all SQLite database operations for MiniVecDB.

    This class follows the "Repository" pattern -- it hides all the
    raw SQL behind clean Python methods. The rest of the codebase
    never writes SQL; it just calls methods like db.insert_record()
    or db.get_metadata().

    Attributes:
        db_path: Filesystem path to the SQLite .db file.

    Example:
        db = DatabaseManager("./data/minivecdb.db")
        db.insert_record("vec_001", "Hello world", "default", time.time())
        row = db.get_record("vec_001")   # -> ("vec_001", "Hello world", ...)
        db.close()
    """

    def __init__(self, db_path: str) -> None:
        """
        Open a SQLite connection and create tables if they don't exist.

        What happens step-by-step:
          1. sqlite3.connect() opens (or creates) the database file.
          2. PRAGMA foreign_keys = ON enables cascade deletes.
          3. executescript(SCHEMA_SQL) creates the three tables + indexes.

        Args:
            db_path: Path to the SQLite database file.
                     Example: "./storage/minivecdb.db"
                     If the file doesn't exist, SQLite creates it.
        """
        # Store the path so other code can inspect it later.
        self.db_path: str = db_path

        # --- Open the connection ---
        # check_same_thread=False lets multiple threads share one
        # connection.  By default SQLite only allows the thread that
        # created the connection to use it.  We turn that off because
        # Flask (our web server) handles requests on different threads.
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
        )

        # --- Enable foreign-key enforcement ---
        # SQLite *parses* FOREIGN KEY clauses but IGNORES them unless
        # you explicitly flip this pragma ON.  Without it, ON DELETE
        # CASCADE won't fire, and you could insert a record pointing
        # at a collection that doesn't exist.
        self._conn.execute("PRAGMA foreign_keys = ON")

        # --- Create tables & indexes ---
        # executescript() can run multiple SQL statements separated by
        # semicolons.  The "IF NOT EXISTS" guards in the schema make
        # this safe to run every time we open the database -- it won't
        # drop or overwrite existing data.
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    @staticmethod
    def _require_non_empty_string(value: str, field_name: str) -> None:
        """Validate that a field is a non-empty string."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string.")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """
        Execute multiple operations inside one SQLite transaction.

        If any operation raises an exception, all writes are rolled back.
        This keeps SQLite data consistent with vector/id-mapping writes.

        Yields:
            None. Use this as a context manager:
            with db.transaction():
                ...
        """
        try:
            self._conn.execute("BEGIN")
            yield
        except Exception:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    # ===============================================================
    # RECORD CRUD  (Create / Read / Update / Delete)
    # ===============================================================
    #
    # The "records" table stores one row per text that was embedded.
    # Each row has:
    #   id          TEXT PRIMARY KEY   -- e.g. "vec_a1b2c3d4"
    #   text        TEXT NOT NULL      -- the original text
    #   collection  TEXT NOT NULL      -- which collection it's in
    #   created_at  REAL NOT NULL      -- Unix timestamp
    #
    # ===============================================================

    def insert_record(
        self,
        id: str,
        text: str,
        collection: str,
        created_at: float,
        auto_commit: bool = True,
    ) -> None:
        """
        Insert a new record into the records table.

        Uses the parameterised query from SQL_QUERIES["insert_record"]:
            INSERT INTO records (id, text, collection, created_at)
            VALUES (?, ?, ?, ?)

        The "?" placeholders are filled in by the tuple we pass as the
        second argument.  This is called a *parameterised query* and is
        the #1 defence against SQL injection attacks.  Never build SQL
        with f-strings or .format() -- always use "?".

        Args:
            id:         Unique record ID  (e.g. "vec_a1b2c3d4").
            text:       The original text that was embedded.
            collection: Which collection this record belongs to.
            created_at: Unix timestamp (seconds since 1970-01-01).
            auto_commit: If True, commit immediately after insert.
                Set to False when this call is part of a larger
                transaction managed by transaction().

        Raises:
            sqlite3.IntegrityError: If a record with this ID already
                exists, or if the collection doesn't exist in the
                collections table (foreign-key violation).
        """
        self._require_non_empty_string(id, "id")
        self._require_non_empty_string(text, "text")
        self._require_non_empty_string(collection, "collection")

        try:
            self._conn.execute(
                SQL_QUERIES["insert_record"],
                (id, text, collection, created_at),
            )
        except sqlite3.Error as exc:
            raise ValueError(f"Failed to insert record '{id}': {exc}") from exc

        if auto_commit:
            self._conn.commit()

    def get_record(self, id: str) -> Optional[Tuple]:
        """
        Fetch a single record by its primary key.

        Args:
            id: The record ID to look up.

        Returns:
            A tuple (id, text, collection, created_at) if found,
            or None if no record has that ID.

        Why a tuple?
            SQLite's cursor.fetchone() returns a tuple by default.
            We could convert it to a dict or dataclass here, but
            keeping it as a tuple is simpler and faster.  The caller
            (usually VectorStore) knows the column order from the
            SELECT clause in SQL_QUERIES["get_record"].
        """
        self._require_non_empty_string(id, "id")
        cursor = self._conn.execute(SQL_QUERIES["get_record"], (id,))
        return cursor.fetchone()

    def delete_record(self, id: str) -> bool:
        """
        Delete a record by ID.

        Because the metadata table has:
            FOREIGN KEY (record_id) REFERENCES records(id) ON DELETE CASCADE
        SQLite will automatically delete all metadata rows that
        reference this record.  You don't need to clean up metadata
        manually -- the database handles it.

        Args:
            id: The record ID to delete.

        Returns:
            True  if a record was actually deleted,
            False if no record with that ID existed.

        How we know if something was deleted:
            cursor.rowcount tells us how many rows the DELETE affected.
            0 means the ID didn't exist; 1 means we deleted it.
        """
        self._require_non_empty_string(id, "id")
        cursor = self._conn.execute(SQL_QUERIES["delete_record"], (id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def update_record_text(self, id: str, new_text: str) -> bool:
        """
        Update the text field of an existing record.

        This does NOT re-embed the text.  The caller (VectorStore) is
        responsible for updating the vector if the text changes.

        Args:
            id:       The record ID to update.
            new_text: The replacement text.

        Returns:
            True  if the record was updated,
            False if no record with that ID existed.
        """
        self._require_non_empty_string(id, "id")
        self._require_non_empty_string(new_text, "new_text")

        # Note the parameter order: the SQL is
        #   UPDATE records SET text=? WHERE id=?
        # so new_text comes FIRST, then id.
        cursor = self._conn.execute(
            SQL_QUERIES["update_record_text"],
            (new_text, id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def record_exists(self, id: str) -> bool:
        """
        Check whether a record with the given ID exists.

        This is faster than get_record() because the query uses
        "SELECT 1 ... LIMIT 1" -- it doesn't fetch any real columns,
        just checks if at least one row matches.

        Args:
            id: The record ID to check.

        Returns:
            True if the record exists, False otherwise.
        """
        self._require_non_empty_string(id, "id")
        cursor = self._conn.execute(SQL_QUERIES["record_exists"], (id,))
        # fetchone() returns (1,) if the row exists, None otherwise.
        return cursor.fetchone() is not None

    # ===============================================================
    # METADATA OPERATIONS  (Entity-Attribute-Value pattern)
    # ===============================================================
    #
    # The "metadata" table uses the EAV pattern:
    #
    #   Entity    = record_id   (which record this tag belongs to)
    #   Attribute = key         (e.g. "category", "author", "year")
    #   Value     = value       (e.g. "science", "Einstein", "2024")
    #
    # WHY EAV?
    #   Different records can have completely different tags.  One
    #   record might have {"category": "science"} while another has
    #   {"language": "french", "topic": "cooking"}.  With a fixed
    #   schema you'd need columns for every possible key.  EAV lets
    #   you add arbitrary keys without altering the table.
    #
    # ===============================================================

    def insert_metadata(
        self,
        record_id: str,
        key: str,
        value: str,
        auto_commit: bool = True,
    ) -> None:
        """
        Attach a key-value metadata tag to a record.

        Args:
            record_id: The record this metadata belongs to.
            key:       Metadata key   (e.g. "category").
            value:     Metadata value (e.g. "science").
            auto_commit: If True, commit immediately after insert.
                Set to False when this call is part of a larger
                transaction managed by transaction().

        Raises:
            sqlite3.IntegrityError: If record_id doesn't exist in the
                records table (foreign-key violation).
        """
        self._require_non_empty_string(record_id, "record_id")
        self._require_non_empty_string(key, "key")

        try:
            self._conn.execute(
                SQL_QUERIES["insert_metadata"],
                (record_id, key, str(value)),
            )
        except sqlite3.Error as exc:
            raise ValueError(
                f"Failed to insert metadata '{key}' for record '{record_id}': {exc}"
            ) from exc

        if auto_commit:
            self._conn.commit()

    def get_metadata(self, record_id: str) -> Dict[str, str]:
        """
        Get all metadata for a record, returned as a dictionary.

        The raw SQL returns rows like:
            [("category", "science"), ("author", "Einstein")]
        We convert that into:
            {"category": "science", "author": "Einstein"}

        Args:
            record_id: The record to get metadata for.

        Returns:
            Dict mapping keys to values.
            Returns an empty dict {} if the record has no metadata.
        """
        self._require_non_empty_string(record_id, "record_id")
        cursor = self._conn.execute(
            SQL_QUERIES["get_metadata"], (record_id,)
        )
        # cursor.fetchall() returns a list of (key, value) tuples.
        # dict() turns that list of 2-tuples directly into a dict.
        return dict(cursor.fetchall())

    def delete_metadata(self, record_id: str) -> None:
        """
        Delete ALL metadata key-value pairs for a given record.

        Useful when you want to replace a record's tags: first delete
        all existing tags, then insert the new ones.

        Args:
            record_id: The record whose metadata to wipe.
        """
        self._require_non_empty_string(record_id, "record_id")
        self._conn.execute(
            SQL_QUERIES["delete_metadata"], (record_id,)
        )
        self._conn.commit()

    def filter_by_metadata(self, filters: Dict[str, str]) -> List[str]:
        """
        Find record IDs that match ALL given metadata filters (AND logic).

        Algorithm:
          1. For each (key, value) pair in filters, run a SELECT to get
             all record_ids that have that exact key=value tag.
          2. Intersect all result sets using Python set intersection.
          3. A record is included only if it appears in EVERY result set
             (i.e. it has ALL the requested tags).

        Example:
            filters = {"category": "science", "year": "2024"}
            Step 1: category=science  -> {"vec_001", "vec_003", "vec_007"}
                    year=2024         -> {"vec_003", "vec_005", "vec_007"}
            Step 2: intersection      -> {"vec_003", "vec_007"}

        Args:
            filters: Dict of key-value pairs to match.
                     Example: {"category": "science", "year": "2024"}

        Returns:
            List of record IDs matching all filters.
            Returns [] if filters is empty or nothing matches.
        """
        if not filters:
            return []

        # Collect one set of record IDs per filter criterion.
        result_sets: List[set] = []

        for key, value in filters.items():
            cursor = self._conn.execute(
                SQL_QUERIES["filter_by_metadata"],
                (key, str(value)),
            )
            # Build a set of IDs from the query results.
            ids = {row[0] for row in cursor.fetchall()}
            result_sets.append(ids)

        # Intersect all sets: start with the first, then AND with each.
        matching_ids = result_sets[0]
        for s in result_sets[1:]:
            matching_ids = matching_ids & s  # set intersection

        return sorted(matching_ids)

    # ===============================================================
    # RECORD LISTING / ID RETRIEVAL
    # ===============================================================

    def get_record_ids_in_collection(self, collection: str) -> List[str]:
        """
        Get all record IDs in a specific collection, ordered by
        creation time (oldest first).

        Args:
            collection: The collection name to list.

        Returns:
            List of record ID strings, e.g. ["vec_001", "vec_002"].
        """
        self._require_non_empty_string(collection, "collection")
        cursor = self._conn.execute(
            SQL_QUERIES["collection_record_ids"], (collection,)
        )
        # Each row is a 1-tuple like ("vec_001",).  We extract element [0].
        return [row[0] for row in cursor.fetchall()]

    def get_all_record_ids(self) -> List[str]:
        """
        Get every record ID across all collections, ordered by
        creation time (oldest first).

        Returns:
            List of all record ID strings.
        """
        cursor = self._conn.execute(SQL_QUERIES["all_record_ids"])
        return [row[0] for row in cursor.fetchall()]

    def get_all_records_with_text(self) -> List[Tuple[str, str]]:
        """
        Get every record's ID and text, ordered by creation time.

        Used by VectorStore._rebuild_vectors() to re-embed all texts
        when the .npy file is corrupted or missing.

        Returns:
            List of (id, text) tuples.
        """
        cursor = self._conn.execute(SQL_QUERIES["all_records_with_text"])
        return cursor.fetchall()

    def list_record_ids(self, collection: Optional[str] = None, limit: int = 100) -> List[str]:
        """
        Get record IDs with an optional collection filter and limit.

        Args:
            collection: If given, only return IDs from this collection.
            limit:      Maximum number of IDs to return.

        Returns:
            List of record ID strings.
        """
        if collection is not None:
            self._require_non_empty_string(collection, "collection")
            cursor = self._conn.execute(
                SQL_QUERIES["list_record_ids_in_collection"],
                (collection, limit),
            )
        else:
            cursor = self._conn.execute(
                SQL_QUERIES["list_record_ids"], (limit,)
            )
        return [row[0] for row in cursor.fetchall()]

    def delete_records_in_collection(self, collection: str) -> int:
        """
        Delete ALL records in a specific collection.

        Cascade deletes also remove metadata rows for those records.

        Args:
            collection: The collection whose records to delete.

        Returns:
            Number of records deleted.
        """
        self._require_non_empty_string(collection, "collection")
        cursor = self._conn.execute(
            SQL_QUERIES["delete_records_in_collection"], (collection,)
        )
        self._conn.commit()
        return cursor.rowcount

    def delete_all_records(self) -> int:
        """
        Delete ALL records across all collections.

        Cascade deletes also remove all metadata rows.

        Returns:
            Number of records deleted.
        """
        cursor = self._conn.execute(SQL_QUERIES["delete_all_records"])
        self._conn.commit()
        return cursor.rowcount

    # ===============================================================
    # COLLECTION CRUD
    # ===============================================================
    #
    # Collections are like folders -- they group related records.
    # The "default" collection is auto-created by the schema.
    #
    # ===============================================================

    def create_collection(
        self,
        name: str,
        dimension: int = 384,
        description: str = "",
    ) -> None:
        """
        Create a new collection.

        Args:
            name:        Unique collection name (e.g. "science_papers").
            dimension:   Vector dimension (384 for all-MiniLM-L6-v2).
            description: Optional human-readable description.

        Raises:
            sqlite3.IntegrityError: If a collection with this name
                already exists (name is the PRIMARY KEY).
        """
        self._require_non_empty_string(name, "name")
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer.")

        try:
            self._conn.execute(
                SQL_QUERIES["create_collection"],
                (name, dimension, description, time.time()),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise ValueError(f"Failed to create collection '{name}': {exc}") from exc

    def list_collections(self) -> List[Tuple]:
        """
        List all collections with their record counts.

        The SQL joins collections with records to compute counts:
            SELECT c.name, c.dimension, c.description, c.created_at,
                   COUNT(r.id) as cnt
            FROM collections c LEFT JOIN records r ...

        Returns:
            List of tuples, each containing:
                (name, dimension, description, created_at, record_count)
        """
        cursor = self._conn.execute(SQL_QUERIES["list_collections"])
        return cursor.fetchall()

    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection.

        Because records have a foreign key to collections with
        ON DELETE CASCADE, deleting a collection also deletes all
        records in it -- and their metadata (cascading further).

        Args:
            name: Collection name to delete.

        Returns:
            True if deleted, False if the collection didn't exist.
        """
        self._require_non_empty_string(name, "name")
        cursor = self._conn.execute(
            SQL_QUERIES["delete_collection"], (name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def collection_exists(self, name: str) -> bool:
        """
        Check whether a collection with the given name exists.

        Args:
            name: Collection name to check.

        Returns:
            True if it exists, False otherwise.
        """
        self._require_non_empty_string(name, "name")
        cursor = self._conn.execute(
            SQL_QUERIES["collection_exists"], (name,)
        )
        return cursor.fetchone() is not None

    # ===============================================================
    # STATISTICS
    # ===============================================================

    def count_records(self, collection: Optional[str] = None) -> int:
        """
        Count records, optionally filtered by collection.

        Args:
            collection: If given, count only records in this collection.
                        If None, count ALL records across all collections.

        Returns:
            Integer count of matching records.
        """
        if collection is None:
            # Use the "count_all_records" query (no WHERE clause).
            cursor = self._conn.execute(SQL_QUERIES["count_all_records"])
        else:
            self._require_non_empty_string(collection, "collection")
            # Use the "count_records" query (WHERE collection=?).
            cursor = self._conn.execute(
                SQL_QUERIES["count_records"], (collection,)
            )
        # COUNT(*) always returns exactly one row with one column.
        return cursor.fetchone()[0]

    def stats_per_collection(self) -> Dict[str, int]:
        """
        Get the record count for each collection.

        The SQL groups records by collection and counts them:
            SELECT collection, COUNT(*) FROM records GROUP BY collection

        Returns:
            Dict mapping collection name to record count.
            Example: {"default": 42, "science": 15}
        """
        cursor = self._conn.execute(SQL_QUERIES["stats_per_collection"])
        # Each row is (collection_name, count).  dict() converts that.
        return dict(cursor.fetchall())

    # ===============================================================
    # CONNECTION MANAGEMENT
    # ===============================================================

    def close(self) -> None:
        """
        Close the SQLite connection and release the file lock.

        Always call this when you're done with the database.  After
        close(), no further operations are possible on this instance.
        """
        self._conn.close()
