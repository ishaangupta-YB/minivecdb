"""
+===============================================================+
|  MiniVecDB -- File Processor Search Integration Tests          |
|  File: tests/test_file_processor_search_integration.py         |
|                                                                |
|  Ensures uploaded tabular rows can be inserted and searched    |
|  without runtime breakage.                                     |
+===============================================================+
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.embeddings import SimpleEmbeddingEngine
from core.file_processor import process_file
from core.vector_store import VectorStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a small deterministic VectorStore for ingestion tests."""
    storage_dir = str(tmp_path / "ingest_search_store")
    store = VectorStore(
        storage_path=storage_dir,
        collection_name="default",
        dimension=384,
    )
    store.embedding_engine = SimpleEmbeddingEngine(dimension=384)
    yield store
    store.db.close()


def test_csv_process_insert_and_search_roundtrip(tmp_path, tmp_store) -> None:
    """Rows produced by process_file() should be searchable after insert_batch()."""
    csv_content = "\n".join(
        [
            "ID,Notes,category",
            "1,minivecdb_upload_probe_xyz alpha token,upload_test",
            "2,ordinary content row,general",
        ]
    )
    file_path = tmp_path / "roundtrip.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "roundtrip.csv",
        max_chars=500,
        overlap=50,
    )
    inserted_ids = tmp_store.insert_batch(texts=texts, metadata_list=metadata_list)

    assert len(inserted_ids) == 2

    results = tmp_store.search("minivecdb_upload_probe_xyz", top_k=3)
    assert len(results) > 0
    assert "minivecdb_upload_probe_xyz" in results[0].record.text


def test_csv_filter_search_after_ingestion(tmp_path, tmp_store) -> None:
    """Metadata filters should still work on rows imported via process_file()."""
    csv_content = "\n".join(
        [
            "ID,Notes,category",
            "1,alpha beta gamma,upload_test",
            "2,beta gamma delta,general",
        ]
    )
    file_path = tmp_path / "filterable.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "filterable.csv",
        max_chars=500,
        overlap=50,
    )
    tmp_store.insert_batch(texts=texts, metadata_list=metadata_list)

    filtered = tmp_store.search(
        "beta",
        top_k=5,
        filters={"category": "upload_test"},
    )
    assert len(filtered) == 1
    assert filtered[0].record.metadata.get("category") == "upload_test"


def test_myxl_like_delayed_header_searchable(tmp_path, tmp_store) -> None:
    """
    myxl-style CSV shape should parse and remain searchable.

    This protects against regressions where preamble lines become headers.
    """
    csv_content = "\n".join(
        [
            "B.TECH - CSE - Section B (2024-28),,,,,",
            ",,,,,",
            "S. No,Name,Enrollment No,Session,Mail ID",
            "1,DISHA JAIN,6417702724,2024-2028,disajain6567@gmail.com",
            "2,RONAK GUPTA,6517702724,2024-2028,ronakgupta3305@gmail.com",
        ]
    )
    file_path = tmp_path / "myxl_like.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "myxl_like.csv",
        max_chars=500,
        overlap=50,
    )
    tmp_store.insert_batch(texts=texts, metadata_list=metadata_list)

    results = tmp_store.search("DISHA JAIN", top_k=3)
    assert len(results) > 0
    assert any("Name: DISHA JAIN" in result.record.text for result in results)
