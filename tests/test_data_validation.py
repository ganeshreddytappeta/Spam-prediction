"""
Unit Tests for Dataset Loading, Column Detection, and Preprocessing (`src/data_loader.py`).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    DatasetError,
    detect_columns,
    load_and_preprocess_dataset,
    normalize_label_value,
    read_csv_safely,
)


def test_normalize_supported_label_representations() -> None:
    # spam / ham
    assert normalize_label_value("spam") == 1
    assert normalize_label_value("HAM") == 0
    # 1 / 0
    assert normalize_label_value(1) == 1
    assert normalize_label_value(0) == 0
    assert normalize_label_value("1") == 1
    assert normalize_label_value("0") == 0
    # spam / not spam
    assert normalize_label_value("not spam") == 0
    # junk / legitimate
    assert normalize_label_value("junk") == 1
    assert normalize_label_value("legitimate") == 0
    # invalid
    assert normalize_label_value("unknown_category") is None
    assert normalize_label_value(None) is None


def test_detect_columns_standard_and_kaggle_names() -> None:
    df1 = pd.DataFrame({"email": ["Hello"], "label": ["ham"]})
    assert detect_columns(df1) == ("email", "label")

    df2 = pd.DataFrame({"v1": ["spam"], "v2": ["Win free cash"]})
    assert detect_columns(df2) == ("v2", "v1")

    df3 = pd.DataFrame({"message": ["Hi"], "category": ["not spam"]})
    assert detect_columns(df3) == ("message", "category")


def test_detect_columns_ambiguous_or_missing_raises_error() -> None:
    # Ambiguous text columns
    ambiguous_df = pd.DataFrame({"text": ["a"], "message": ["b"], "label": ["ham"]})
    with pytest.raises(DatasetError, match="Ambiguous email text columns"):
        detect_columns(ambiguous_df)

    # Missing text column
    missing_df = pd.DataFrame({"col_a": ["a"], "label": ["ham"]})
    with pytest.raises(DatasetError, match="Could not automatically detect an email text column"):
        detect_columns(missing_df)

    # Missing label column when required
    no_label_df = pd.DataFrame({"text": ["hello world"]})
    with pytest.raises(DatasetError, match="Could not automatically detect a classification label column"):
        detect_columns(no_label_df, require_label=True)


def test_missing_and_empty_csv_files(tmp_path: Path) -> None:
    missing_file = tmp_path / "nonexistent.csv"
    with pytest.raises(DatasetError, match="Dataset file not found"):
        read_csv_safely(missing_file)

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="empty"):
        read_csv_safely(empty_file)


def test_dataset_cleaning_duplicates_missing_and_invalid_labels(tmp_path: Path) -> None:
    csv_file = tmp_path / "custom_emails.csv"
    raw_data = pd.DataFrame(
        {
            "body": [
                "Legitimate meeting agenda for Monday morning",
                "Legitimate meeting agenda for Monday morning",  # duplicate
                "WIN FREE CASH PRIZE NOW!!!",
                "Claim your junk lottery reward",
                "   ",  # empty whitespace message
                None,   # missing text
                "Valid text with invalid label",
            ],
            "category": [
                "not spam",
                "not spam",
                "spam",
                "junk",
                "ham",
                "spam",
                "invalid_label_xyz",
            ],
        }
    )
    raw_data.to_csv(csv_file, index=False)

    cleaned_df, stats = load_and_preprocess_dataset(file_path=csv_file)
    assert stats["initial_rows"] == 7
    assert stats["missing_text_count"] == 1
    assert stats["empty_messages_removed"] == 1
    assert stats["invalid_labels_removed"] == 1
    assert stats["duplicate_records_removed"] == 1
    assert stats["final_rows"] == 3
    assert stats["spam_count"] == 2
    assert stats["ham_count"] == 1
    assert list(cleaned_df.columns) == ["text", "label", "label_name"]

