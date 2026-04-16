"""
+===============================================================+
|  MiniVecDB -- File Processor Tests                             |
|  File: tests/test_file_processor.py                            |
|                                                                |
|  Validates CSV/Excel/TXT ingestion behavior for the shared     |
|  upload pipeline in core/file_processor.py.                    |
+===============================================================+
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.file_processor import process_file


def test_process_file_csv_detects_delayed_header_and_serializes_rows(tmp_path) -> None:
    """
    A myxl-like CSV with preamble lines should detect the real header row.

    Expected behavior:
      - preamble/title lines are ignored
      - each student row is serialized with column context
      - short rows remain single chunks
    """
    csv_content = "\n".join(
        [
            "B.TECH - CSE - Section B (2024-28),,,,,",
            ",,,,,,",
            "S. No,Name,Enrollment No,Programme Name,Session,Mail ID",
            "1,DISHA JAIN,6417702724,Computer Science Engineering,2024-2028,disajain6567@gmail.com",
            "2,RONAK GUPTA,6517702724,Computer Science Engineering,2024-2028,ronakgupta3305@gmail.com",
        ]
    )
    file_path = tmp_path / "class.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "class.csv",
        max_chars=500,
        overlap=50,
    )

    assert len(texts) == 2
    assert "Name: DISHA JAIN" in texts[0]
    assert "Enrollment No: 6417702724" in texts[0]
    assert "Mail ID: disajain6567@gmail.com" in texts[0]
    assert all("S. No: S. No" not in text for text in texts)

    assert [m["row"] for m in metadata_list] == ["1", "2"]
    assert all(m["chunk_index"] == "0" for m in metadata_list)
    assert all(m["total_chunks"] == "1" for m in metadata_list)


def test_process_file_csv_short_rows_are_single_chunks(tmp_path) -> None:
    """Rows below max_chars should produce exactly one chunk each."""
    csv_content = "\n".join(
        [
            "Name,Department,Roll",
            "Aman,CSE,1",
            "Isha,ECE,2",
            "Nora,ME,3",
        ]
    )
    file_path = tmp_path / "short_rows.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "short_rows.csv",
        max_chars=500,
        overlap=50,
    )

    assert len(texts) == 3
    assert all(meta["chunk_index"] == "0" for meta in metadata_list)
    assert all(meta["total_chunks"] == "1" for meta in metadata_list)


def test_process_file_csv_long_row_splits_without_overlap(tmp_path) -> None:
    """
    Long serialized tabular rows should split with zero overlap.

    This verifies:
      - chunks are bounded by max_chars
      - all chunks stay tied to the same row metadata
      - neighbouring chunks are not overlap-seeded
    """
    long_value = " ".join(f"token_{idx:03d}" for idx in range(220))
    csv_content = "\n".join(
        [
            "Name,Description",
            f"Alpha,{long_value}",
        ]
    )
    file_path = tmp_path / "long_row.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "long_row.csv",
        max_chars=120,
        overlap=50,
    )

    assert len(texts) > 1
    assert all(len(chunk) <= 120 for chunk in texts)
    assert all(meta["row"] == "1" for meta in metadata_list)

    expected_total = str(len(texts))
    for idx, meta in enumerate(metadata_list):
        assert meta["chunk_index"] == str(idx)
        assert meta["total_chunks"] == expected_total

    for left, right in zip(texts, texts[1:]):
        assert not right.startswith(left[-20:])


def test_process_file_excel_detects_delayed_header(tmp_path) -> None:
    """Excel files with preamble rows should resolve the true header row."""
    df = pd.DataFrame(
        [
            ["B.TECH - CSE - Section B (2024-28)", "", "", "", ""],
            ["", "", "", "", ""],
            ["S. No", "Name", "Enrollment No", "Session", "Mail ID"],
            ["1", "DISHA JAIN", "6417702724", "2024-2028", "disajain6567@gmail.com"],
        ]
    )
    file_path = tmp_path / "class.xlsx"
    df.to_excel(file_path, index=False, header=False)

    texts, metadata_list = process_file(
        str(file_path),
        "class.xlsx",
        max_chars=500,
        overlap=50,
        sheet_name=0,
    )

    assert len(texts) == 1
    assert "Name: DISHA JAIN" in texts[0]
    assert "Enrollment No: 6417702724" in texts[0]
    assert metadata_list[0]["sheet"] == "0"
    assert metadata_list[0]["chunk_index"] == "0"
    assert metadata_list[0]["total_chunks"] == "1"


def test_process_file_header_row_override(tmp_path) -> None:
    """Explicit header_row should override auto-detection when needed."""
    csv_content = "\n".join(
        [
            "metadata,value",
            "ColA,ColB",
            "one,two",
        ]
    )
    file_path = tmp_path / "override.csv"
    file_path.write_text(csv_content, encoding="utf-8")

    texts, metadata_list = process_file(
        str(file_path),
        "override.csv",
        max_chars=500,
        overlap=50,
        header_row=1,
    )

    assert len(texts) == 1
    assert "ColA: one" in texts[0]
    assert "ColB: two" in texts[0]
    assert metadata_list[0]["row"] == "1"
