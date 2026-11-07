"""
Prediction and Inference Module for MailGuard AI.

Provides safe model loading, single-email prediction with probability estimates
and signal analysis, and bulk CSV DataFrame classification. Supports both full
Scikit-learn/Joblib pipelines (`models/best_model.joblib`) and portable JSON
model weights (`models/model_weights.json`) for lightweight serverless deployment.
"""

from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

from src.preprocessing import extract_email_signals, preprocess_text

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


class ModelLoadError(Exception):
    """Raised when the trained model or metadata file is missing or corrupted."""


class PredictionInputError(Exception):
    """Raised when input email text or bulk DataFrame fails validation."""


def _resolve_model_paths() -> Tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    return (
        models_dir / "best_model.joblib",
        models_dir / "model_metadata.json",
        models_dir / "model_weights.json",
    )


class PortablePipeline:
    """
    Mathematically identical pure-Python execution engine for the trained
    `EmailTextPreprocessor` + `TfidfVectorizer` + `LogisticRegression` / `MultinomialNB`
    pipeline when deployed to serverless environments without C-extensions.
    """

    def __init__(self, weights: Dict[str, Any]) -> None:
        self.vocab: Dict[str, int] = weights["vocabulary"]
        self.idf: List[float] = weights["idf"]
        self.ngram_range: Tuple[int, int] = tuple(weights.get("ngram_range", [1, 2]))  # type: ignore[assignment]
        self.sublinear_tf: bool = bool(weights.get("sublinear_tf", True))
        self.classifier_type: str = weights.get("classifier_type", "logistic_regression")
        self.coef: List[float] = weights.get("coef", [])
        self.intercept: float = float(weights.get("intercept", 0.0))
        self.feature_log_prob_ham: List[float] = weights.get("feature_log_prob_ham", [])
        self.feature_log_prob_spam: List[float] = weights.get("feature_log_prob_spam", [])
        self.class_log_prior: List[float] = weights.get("class_log_prior", [0.0, 0.0])

    def _vectorize_single(self, raw_text: str) -> Dict[int, float]:
        cleaned = preprocess_text(raw_text, remove_stopwords=True)
        tokens = _TOKEN_PATTERN.findall(cleaned.lower())
        min_n, max_n = self.ngram_range
        ngrams: List[str] = []
        n_tokens = len(tokens)
        for n in range(min_n, min(max_n + 1, n_tokens + 1)):
            if n == 1:
                ngrams.extend(tokens)
            else:
                ngrams.extend(" ".join(tokens[i : i + n]) for i in range(n_tokens - n + 1))

        counts = Counter(ng for ng in ngrams if ng in self.vocab)
        if not counts:
            return {}

        vec: Dict[int, float] = {}
        norm_sq = 0.0
        for term, count in counts.items():
            idx = self.vocab[term]
            tf = (1.0 + math.log(count)) if self.sublinear_tf else float(count)
            val = tf * self.idf[idx]
            vec[idx] = val
            norm_sq += val * val

        if norm_sq > 0.0:
            norm = math.sqrt(norm_sq)
            for idx in vec:
                vec[idx] /= norm
        return vec

    def predict_proba(self, X: Iterable[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for doc in X:
            vec = self._vectorize_single(str(doc))
            if self.classifier_type == "logistic_regression":
                z = self.intercept + sum(val * self.coef[idx] for idx, val in vec.items())
                if z >= 0:
                    spam_prob = 1.0 / (1.0 + math.exp(-z))
                else:
                    ez = math.exp(z)
                    spam_prob = ez / (1.0 + ez)
                ham_prob = 1.0 - spam_prob
                results.append([ham_prob, spam_prob])
            else:
                jll_ham = self.class_log_prior[0] + sum(
                    val * self.feature_log_prob_ham[idx] for idx, val in vec.items()
                )
                jll_spam = self.class_log_prior[1] + sum(
                    val * self.feature_log_prob_spam[idx] for idx, val in vec.items()
                )
                max_jll = max(jll_ham, jll_spam)
                eh = math.exp(jll_ham - max_jll)
                es = math.exp(jll_spam - max_jll)
                total = eh + es
                results.append([eh / total, es / total])
        return results

    def predict(self, X: Iterable[str]) -> List[int]:
        probs = self.predict_proba(X)
        return [1 if p[1] >= 0.5 else 0 for p in probs]


def load_trained_model(
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load the persisted Scikit-learn Pipeline and model metadata from `models/`.
    Falls back to `models/model_weights.json` when running in a serverless
    environment without `joblib` / `scikit-learn`.

    Raises
    ------
    ModelLoadError
        If the model file is missing, unreadable, or corrupted.
    """
    default_model_path, default_meta_path, default_weights_path = _resolve_model_paths()
    resolved_model_path = Path(model_path) if model_path else default_model_path
    resolved_meta_path = Path(metadata_path) if metadata_path else default_meta_path

    pipeline: Any = None
    if joblib is not None:
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
    elif model_path is None and default_weights_path.exists():
        try:
            with open(default_weights_path, "r", encoding="utf-8") as f:
                weights = json.load(f)
            pipeline = PortablePipeline(weights)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load portable model weights from '{default_weights_path}': {exc}"
            ) from exc
    else:
        raise ModelLoadError(
            f"Trained model file not found at '{resolved_model_path}'."
        )

    if not hasattr(pipeline, "predict"):
        raise ModelLoadError(
            f"The loaded object from '{resolved_model_path}' is not a valid estimator."
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
    df: Any,
    text_column: str,
    pipeline: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Classify multiple email messages from an uploaded CSV DataFrame.
    Handles empty or missing rows gracefully without crashing.
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
