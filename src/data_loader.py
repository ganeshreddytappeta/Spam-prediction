"""
Data Loading and Dataset Validation Module for MailGuard AI.

Handles reading raw email datasets from CSV files, automatically detecting
email content and label columns, cleaning missing/empty/duplicate records,
normalizing spam/ham labels to 1/0, and recording preprocessing statistics.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Supported candidate column names (lowercase) for automatic detection
TEXT_COLUMN_CANDIDATES: List[str] = [
    "text",
    "email",
    "message",
    "body",
    "content",
    "email_text",
    "mail",
    "sms",
    "v2",
]

LABEL_COLUMN_CANDIDATES: List[str] = [
    "label",
    "category",
    "spam",
    "class",
    "target",
    "type",
    "status",
    "v1",
]

# Standardized mapping from common label representations to binary (Spam=1, Ham=0)
LABEL_MAPPING: Dict[str, int] = {
    "spam": 1,
    "1": 1,
    "1.0": 1,
    "junk": 1,
    "true": 1,
    "yes": 1,
    "malicious": 1,
    "phishing": 1,
    "ham": 0,
    "0": 0,
    "0.0": 0,
    "not spam": 0,
    "not_spam": 0,
    "non-spam": 0,
    "legitimate": 0,
    "safe": 0,
    "normal": 0,
    "false": 0,
    "no": 0,
}


class DatasetError(Exception):
    """Custom exception raised when dataset loading or validation fails."""


def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parent.parent


def get_default_dataset_path() -> Path:
    """
    Resolve the default dataset path.
    Checks `data/emails.csv` first, falling back to `spam.csv` in the project root
    if `data/emails.csv` has not been created yet.
    """
    root = get_project_root()
    primary_path = root / "data" / "emails.csv"
    fallback_path = root / "spam.csv"

    if primary_path.exists():
        return primary_path
    if fallback_path.exists():
        return fallback_path
    return primary_path


def read_csv_safely(file_path_or_buffer: Any) -> pd.DataFrame:
    """
    Read a CSV file or uploaded stream safely across common text encodings.

    Tries UTF-8 first, then UTF-8-SIG, CP1252, and Latin-1 (ISO-8859-1).
    Raises DatasetError if the file is empty, missing, or unreadable.
    """
    if isinstance(file_path_or_buffer, (str, Path)):
        path = Path(file_path_or_buffer)
        if not path.exists():
            raise DatasetError(
                f"Dataset file not found at '{path}'.\n"
                "How to download and place the dataset:\n"
                "1. Download a real labeled Email/SMS Spam CSV dataset from Kaggle "
                "(https://www.kaggle.com/datasets).\n"
                "2. Place the CSV file inside the project at: data/emails.csv"
            )
        if path.stat().st_size == 0:
            raise DatasetError(f"The CSV file at '{path}' is completely empty (0 bytes).")

    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Optional[Exception] = None

    for encoding in encodings_to_try:
        try:
            if hasattr(file_path_or_buffer, "seek"):
                file_path_or_buffer.seek(0)
            df = pd.read_csv(file_path_or_buffer, encoding=encoding)
            if df is None or df.empty or len(df.columns) == 0:
                raise DatasetError("The uploaded or specified CSV file contains no data rows.")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise DatasetError("The CSV file is empty or contains no valid columns.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(f"Malformed CSV format: {exc}") from exc

    raise DatasetError(
        f"Unable to decode CSV file with supported encodings ({', '.join(encodings_to_try)}). "
        f"Last error: {last_error}"
    )


def detect_columns(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    require_label: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Automatically inspect DataFrame columns to identify the email content column
    and the classification label column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame loaded from CSV.
    text_col : str, optional
        Explicitly configured text column name.
    label_col : str, optional
        Explicitly configured label column name.
    require_label : bool, default=True
        Whether a label column is mandatory (True for training, False for bulk inference).

    Returns
    -------
    Tuple[str, Optional[str]]
        (detected_text_column, detected_label_column)

    Raises
    ------
    DatasetError
        If columns are missing or automatic detection is ambiguous.
    """
    if df is None or df.empty:
        raise DatasetError("Cannot detect columns on an empty dataset.")

    available_cols = list(df.columns)
    lower_to_actual: Dict[str, str] = {str(col).strip().lower(): col for col in available_cols}

    # 1. Resolve text column
    resolved_text_col: Optional[str] = None
    if text_col is not None:
        if text_col in available_cols:
            resolved_text_col = text_col
        elif text_col.strip().lower() in lower_to_actual:
            resolved_text_col = lower_to_actual[text_col.strip().lower()]
        else:
            raise DatasetError(
                f"Configured text column '{text_col}' was not found in CSV columns: {available_cols}"
            )
    else:
        matched_text_cols = [
            lower_to_actual[cand]
            for cand in TEXT_COLUMN_CANDIDATES
            if cand in lower_to_actual
        ]
        # Deduplicate while preserving order
        matched_text_cols = list(dict.fromkeys(matched_text_cols))

        if len(matched_text_cols) == 1:
            resolved_text_col = matched_text_cols[0]
        elif len(matched_text_cols) > 1:
            raise DatasetError(
                f"Ambiguous email text columns found: {matched_text_cols}. "
                "Please specify the exact email content column name explicitly."
            )
        else:
            raise DatasetError(
                f"Could not automatically detect an email text column in {available_cols}. "
                f"Supported column names include: {TEXT_COLUMN_CANDIDATES}. "
                "Please rename your column or specify the column configuration."
            )

    # 2. Resolve label column (if required or present)
    resolved_label_col: Optional[str] = None
    if label_col is not None:
        if label_col in available_cols:
            resolved_label_col = label_col
        elif label_col.strip().lower() in lower_to_actual:
            resolved_label_col = lower_to_actual[label_col.strip().lower()]
        else:
            raise DatasetError(
                f"Configured label column '{label_col}' was not found in CSV columns: {available_cols}"
            )
    else:
        matched_label_cols = [
            lower_to_actual[cand]
            for cand in LABEL_COLUMN_CANDIDATES
            if cand in lower_to_actual and lower_to_actual[cand] != resolved_text_col
        ]
        matched_label_cols = list(dict.fromkeys(matched_label_cols))

        if len(matched_label_cols) == 1:
            resolved_label_col = matched_label_cols[0]
        elif len(matched_label_cols) > 1 and require_label:
            raise DatasetError(
                f"Ambiguous classification label columns found: {matched_label_cols}. "
                "Please specify the exact label column name explicitly."
            )
        elif len(matched_label_cols) == 0 and require_label:
            raise DatasetError(
                f"Could not automatically detect a classification label column in {available_cols}. "
                f"Supported label column names include: {LABEL_COLUMN_CANDIDATES}."
            )

    return resolved_text_col, resolved_label_col


def normalize_label_value(val: Any) -> Optional[int]:
    """
    Normalize a raw classification label into 1 (Spam) or 0 (Ham).
    Returns None if the label is missing or unrecognized.
    """
    if pd.isna(val):
        return None
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)) and val in (0, 1):
        return int(val)

    cleaned = str(val).strip().lower()
    return LABEL_MAPPING.get(cleaned, None)


def load_and_preprocess_dataset(
    file_path: Optional[ Any ] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the complete 12-step dataset loading, validation, and preprocessing pipeline:

    1. Load the CSV dataset.
    2. Record initial dataset dimensions.
    3. Inspect column names and data types (and merge overflow unnamed columns if present).
    4. Identify missing values.
    5. Remove missing email content.
    6. Detect duplicate records.
    7. Remove duplicates.
    8. Normalize classification labels (Spam = 1, Ham = 0).
    9. Handle and remove invalid labels.
    10. Check class distribution.
    11. Identify and remove empty/whitespace-only messages.
    12. Prepare the final standardized dataset (`text`, `label`, `label_name`).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Cleaned DataFrame with columns `['text', 'label', 'label_name']`
        - Dictionary of detailed preprocessing statistics
    """
    target_path = file_path if file_path is not None else get_default_dataset_path()

    # Step 1: Load the CSV dataset
    raw_df = read_csv_safely(target_path)

    # Step 2: Display / record dataset dimensions
    initial_rows, initial_cols = raw_df.shape

    # Step 3: Inspect column names and data types
    original_columns = [str(c) for c in raw_df.columns]
    column_dtypes = {str(col): str(dtype) for col, dtype in raw_df.dtypes.items()}

    detected_text_col, detected_label_col = detect_columns(
        raw_df, text_col=text_col, label_col=label_col, require_label=True
    )
    assert detected_label_col is not None

    df = raw_df.copy()

    # Handle trailing 'Unnamed:' columns that sometimes hold spilled text from unquoted commas
    unnamed_cols = [
        c for c in df.columns
        if str(c).startswith("Unnamed:") and c not in (detected_text_col, detected_label_col)
    ]
    if unnamed_cols:
        for u_col in unnamed_cols:
            mask = df[u_col].notna() & (df[u_col].astype(str).str.strip() != "")
            if mask.any():
                df.loc[mask, detected_text_col] = (
                    df.loc[mask, detected_text_col].fillna("").astype(str)
                    + " "
                    + df.loc[mask, u_col].astype(str)
                )
        df = df.drop(columns=unnamed_cols)

    # Step 4: Identify missing values
    missing_text_count = int(df[detected_text_col].isna().sum())
    missing_label_count = int(df[detected_label_col].isna().sum())
    total_missing_values = int(df[[detected_text_col, detected_label_col]].isna().sum().sum())

    # Step 5: Remove missing email content or missing labels
    df = df.dropna(subset=[detected_text_col, detected_label_col]).copy()
    rows_after_missing_removal = len(df)

    # Step 11 (early check + post-strip check): Identify empty messages
    df[detected_text_col] = df[detected_text_col].astype(str)
    empty_message_mask = df[detected_text_col].str.strip() == ""
    empty_messages_count = int(empty_message_mask.sum())
    if empty_messages_count > 0:
        df = df.loc[~empty_message_mask].copy()

    # Step 8 & 9: Normalize classification labels & handle invalid labels
    raw_unique_labels = [str(val) for val in df[detected_label_col].unique()[:20]]
    df["label"] = df[detected_label_col].apply(normalize_label_value)

    invalid_label_mask = df["label"].isna()
    invalid_labels_count = int(invalid_label_mask.sum())
    if invalid_labels_count > 0:
        df = df.loc[~invalid_label_mask].copy()

    if df.empty:
        raise DatasetError(
            f"No valid records remained after normalizing labels in column '{detected_label_col}'. "
            f"Found raw values: {raw_unique_labels}. "
            "Supported labels: spam/ham, 1/0, spam/not spam, junk/legitimate."
        )

    df["label"] = df["label"].astype(int)

    # Step 6 & 7: Detect and remove duplicate records
    duplicate_count = int(df.duplicated(subset=[detected_text_col, "label"]).sum())
    # Also check exact text duplicates
    text_duplicate_count = int(df.duplicated(subset=[detected_text_col]).sum())
    duplicate_records_removed = text_duplicate_count
    df = df.drop_duplicates(subset=[detected_text_col], keep="first").copy()

    # Step 12: Prepare final dataset
    df["text"] = df[detected_text_col].astype(str).str.strip()
    df["label_name"] = df["label"].map({1: "SPAM", 0: "HAM"})
    final_df = df[["text", "label", "label_name"]].reset_index(drop=True)

    # Step 10: Check class distribution
    final_rows = len(final_df)
    spam_count = int((final_df["label"] == 1).sum())
    ham_count = int((final_df["label"] == 0).sum())

    if spam_count == 0 or ham_count == 0:
        raise DatasetError(
            f"Dataset must contain both Spam (1) and Ham (0) examples. "
            f"Found Spam={spam_count}, Ham={ham_count}."
        )

    spam_percentage = round((spam_count / final_rows) * 100, 2) if final_rows > 0 else 0.0
    ham_percentage = round((ham_count / final_rows) * 100, 2) if final_rows > 0 else 0.0

    # Character and word length statistics for analytics
    char_lengths = final_df["text"].str.len()
    word_counts = final_df["text"].str.split().str.len()

    stats: Dict[str, Any] = {
        "dataset_path": str(target_path),
        "initial_rows": initial_rows,
        "initial_columns": initial_cols,
        "original_columns": original_columns,
        "unnamed_columns_merged": unnamed_cols,
        "column_dtypes": column_dtypes,
        "detected_text_column": detected_text_col,
        "detected_label_column": detected_label_col,
        "raw_unique_labels": raw_unique_labels,
        "missing_text_count": missing_text_count,
        "missing_label_count": missing_label_count,
        "total_missing_values": total_missing_values,
        "rows_after_missing_removal": rows_after_missing_removal,
        "empty_messages_removed": empty_messages_count,
        "invalid_labels_removed": invalid_labels_count,
        "exact_duplicates_detected": duplicate_count,
        "duplicate_records_removed": duplicate_records_removed,
        "final_rows": final_rows,
        "final_columns": list(final_df.columns),
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": spam_percentage,
        "ham_percentage": ham_percentage,
        "avg_char_length": round(float(char_lengths.mean()), 1),
        "avg_word_count": round(float(word_counts.mean()), 1),
        "avg_spam_char_length": round(
            float(final_df.loc[final_df["label"] == 1, "text"].str.len().mean()), 1
        ),
        "avg_ham_char_length": round(
            float(final_df.loc[final_df["label"] == 0, "text"].str.len().mean()), 1
        ),
    }

    return final_df, stats

