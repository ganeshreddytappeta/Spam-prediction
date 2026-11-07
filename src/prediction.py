"""
Prediction and Inference Module for MailGuard AI.

Provides safe model loading, single-email prediction with probability estimates
and signal analysis, and bulk CSV DataFrame classification.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

from src.preprocessing import extract_email_signals, preprocess_text
from src.training import get_default_model_paths


class ModelLoadError(Exception):
    """Raised when the trained model or metadata file is missing or corrupted."""


class PredictionInputError(Exception):
    """Raised when input email text or bulk DataFrame fails validation."""


def load_trained_model(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load the persisted Scikit-learn Pipeline and model metadata from `models/`.

    Raises
    ------
    ModelLoadError
        If the model file is missing, unreadable, or corrupted.
    """
    default_model_path, default_meta_path = get_default_model_paths()
    resolved_model_path = Path(model_path) if model_path else default_model_path
    resolved_meta_path = Path(metadata_path) if metadata_path else default_meta_path

    if not resolved_model_path.exists():
        raise ModelLoadError(
            f"Trained model file not found at '{resolved_model_path}'. "
            "Please train the model first by running: python train.py"
        )

    try:
        pipeline = joblib.load(resolved_model_path)
    except Exception as exc:
        raise ModelLoadError(
            f"Failed to load model from '{resolved_model_path}'. The file may be corrupted. "
            f"Please re-run `python train.py` to regenerate the model. Details: {exc}"
        ) from exc

    if not hasattr(pipeline, "predict"):
        raise ModelLoadError(
            f"The loaded object from '{resolved_model_path}' is not a valid Scikit-learn estimator."
        )

    metadata: Dict[str, Any] = {}
    if resolved_meta_path.exists():
        try:
            with open(resolved_meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as exc:
            raise ModelLoadError(
                f"Model metadata file at '{resolved_meta_path}' is corrupted: {exc}. "
                "Please re-run `python train.py`."
            ) from exc

    return pipeline, metadata


def predict_single_email(
    email_text: Any,
    pipeline: Any,
) -> Dict[str, Any]:
    """
    Validate and classify a single email message using the trained pipeline.

    Parameters
    ----------
    email_text : str
        Raw email message entered by the user.
    pipeline : sklearn.pipeline.Pipeline
        Loaded classification pipeline.

    Returns
    -------
    Dict[str, Any]
        Prediction results including label ('SPAM' or 'HAM'), binary class (1 or 0),
        spam/ham probabilities, confidence percentage, preprocessed text, and signals.
    """
    if email_text is None or not isinstance(email_text, str):
        raise PredictionInputError("Email input must be a valid text string.")

    stripped = email_text.strip()
    if not stripped:
        raise PredictionInputError("Email content cannot be empty or contain only whitespace.")

    cleaned = preprocess_text(stripped)
    if not cleaned:
        raise PredictionInputError(
            "Email content contains no recognizable words or tokens after preprocessing."
        )

    pred_class = int(pipeline.predict([stripped])[0])
    pred_label = "SPAM" if pred_class == 1 else "HAM"

    spam_prob: Optional[float] = None
    ham_prob: Optional[float] = None
    confidence: Optional[float] = None

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([stripped])[0]
        ham_prob = round(float(probs[0]), 4)
        spam_prob = round(float(probs[1]), 4)
        confidence = round(float(max(ham_prob, spam_prob) * 100), 2)

    signals = extract_email_signals(stripped)

    return {
        "raw_text": stripped,
        "cleaned_text": cleaned,
        "predicted_class": pred_class,
        "predicted_label": pred_label,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
        "confidence_percent": confidence,
        "signals": signals,
    }


def predict_bulk_emails(
    df: pd.DataFrame,
    text_column: str,
    pipeline: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Classify multiple email messages from an uploaded CSV DataFrame.

    Handles empty or missing rows gracefully without crashing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame from uploaded CSV.
    text_column : str
        Column containing email text.
    pipeline : sklearn.pipeline.Pipeline
        Loaded classification pipeline.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (Result DataFrame with added prediction columns, Summary dictionary)
    """
    if df is None or df.empty:
        raise PredictionInputError("Uploaded DataFrame is empty.")

    if text_column not in df.columns:
        raise PredictionInputError(
            f"Selected text column '{text_column}' does not exist in the uploaded CSV. "
            f"Available columns: {list(df.columns)}"
        )

    result_df = df.copy()
    raw_series = result_df[text_column]

    # Identify valid vs empty/missing messages
    valid_mask = raw_series.notna() & (raw_series.astype(str).str.strip() != "")
    also_valid_tokens = raw_series.fillna("").astype(str).apply(lambda x: bool(preprocess_text(x)))
    valid_mask = valid_mask & also_valid_tokens

    total_rows = len(result_df)
    valid_rows = int(valid_mask.sum())
    empty_rows = total_rows - valid_rows

    if valid_rows == 0:
        raise PredictionInputError(
            f"All {total_rows} rows in column '{text_column}' are empty or contain no valid text."
        )

    valid_texts = result_df.loc[valid_mask, text_column].astype(str).tolist()
    preds = pipeline.predict(valid_texts)

    result_df["predicted_label"] = "EMPTY / INVALID"
    result_df.loc[valid_mask, "predicted_label"] = [
        "SPAM" if int(p) == 1 else "HAM" for p in preds
    ]

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(valid_texts)
        result_df["spam_probability"] = None
        result_df["ham_probability"] = None
        result_df.loc[valid_mask, "spam_probability"] = [round(float(p[1]), 4) for p in probs]
        result_df.loc[valid_mask, "ham_probability"] = [round(float(p[0]), 4) for p in probs]

    spam_count = int((result_df["predicted_label"] == "SPAM").sum())
    ham_count = int((result_df["predicted_label"] == "HAM").sum())

    summary = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "empty_rows": empty_rows,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_percentage": round((spam_count / valid_rows) * 100, 2) if valid_rows > 0 else 0.0,
        "ham_percentage": round((ham_count / valid_rows) * 100, 2) if valid_rows > 0 else 0.0,
    }

    return result_df, summary
