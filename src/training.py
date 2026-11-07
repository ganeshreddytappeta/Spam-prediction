"""
Machine Learning Model Training and Persistence Module for MailGuard AI.

Builds end-to-end Scikit-learn Pipelines (Preprocessing -> TF-IDF -> Classifier)
for Multinomial Naive Bayes and Logistic Regression, prevents data leakage by
splitting data prior to fitting, compares models, extracts top TF-IDF indicators,
and saves the best pipeline (`models/best_model.joblib`) alongside metadata
(`models/model_metadata.json`).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data_loader import get_project_root, load_and_preprocess_dataset
from src.evaluation import evaluate_model, select_best_model
from src.preprocessing import EmailTextPreprocessor


def get_default_model_paths() -> Tuple[Path, Path]:
    """Return the default paths for the saved model pipeline and metadata JSON."""
    root = get_project_root()
    models_dir = root / "models"
    return models_dir / "best_model.joblib", models_dir / "model_metadata.json"


def build_pipeline(classifier: Any, ngram_range: Tuple[int, int] = (1, 2), max_features: int = 6000, min_df: int = 2) -> Pipeline:
    """
    Create a Scikit-learn Pipeline combining custom NLP text preprocessing,
    TF-IDF feature extraction, and the classification estimator.

    Because the vectorizer is inside the Pipeline, it learns its vocabulary
    exclusively from the training split during `pipeline.fit(X_train, y_train)`,
    strictly preventing data leakage from the test dataset.
    """
    return Pipeline(
        steps=[
            ("preprocessor", EmailTextPreprocessor(remove_stopwords=True)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=min_df,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def extract_top_tfidf_features(pipeline: Pipeline, top_n: int = 20) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract the top spam-indicating and ham-indicating terms/n-grams learned by
    the trained TF-IDF + Classifier pipeline.
    """
    try:
        tfidf: TfidfVectorizer = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["classifier"]
        feature_names = np.array(tfidf.get_feature_names_out())

        if hasattr(clf, "coef_"):
            # Logistic Regression: positive coefficients indicate Spam (class 1)
            weights = clf.coef_[0]
        elif hasattr(clf, "feature_log_prob_"):
            # MultinomialNB: log P(word | Spam) - log P(word | Ham)
            weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
        else:
            return {"top_spam_features": [], "top_ham_features": []}

        top_spam_idx = np.argsort(weights)[::-1][:top_n]
        top_ham_idx = np.argsort(weights)[:top_n]

        top_spam = [
            {"feature": str(feature_names[idx]), "weight": round(float(weights[idx]), 4)}
            for idx in top_spam_idx
        ]
        top_ham = [
            {"feature": str(feature_names[idx]), "weight": round(float(abs(weights[idx])), 4)}
            for idx in top_ham_idx
        ]
        return {"top_spam_features": top_spam, "top_ham_features": top_ham}
    except Exception:
        return {"top_spam_features": [], "top_ham_features": []}


def train_and_save_models(
    dataset_path: Optional[Any] = None,
    model_output_path: Optional[Path] = None,
    metadata_output_path: Optional[Path] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute the complete model development workflow:
    1. Load and preprocess the dataset.
    2. Perform a stratified 80:20 train/test split before vectorization.
    3. Train Multinomial Naive Bayes and Logistic Regression pipelines.
    4. Evaluate both models on the held-out test set.
    5. Select the best model using Spam F1, Precision, Recall, and overall metrics.
    6. Save the winning pipeline (`best_model.joblib`) and metadata (`model_metadata.json`).
    """
    default_model_path, default_meta_path = get_default_model_paths()
    model_path = Path(model_output_path) if model_output_path else default_model_path
    meta_path = Path(metadata_output_path) if metadata_output_path else default_meta_path

    # 1. Load and preprocess dataset
    df, preprocessing_stats = load_and_preprocess_dataset(file_path=dataset_path)

    X = df["text"].tolist()
    y = df["label"].to_numpy()

    # 2. Stratified 80:20 train-test split BEFORE fitting TF-IDF
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    tfidf_config = {
        "ngram_range": [1, 2],
        "max_features": 6000,
        "min_df": 2,
        "sublinear_tf": True,
        "explanation": (
            "Unigrams and bigrams (1, 2) with max_features=6000, min_df=2, and "
            "sublinear_tf=True capture both individual spam keywords ('free', 'claim') "
            "and multi-word promotional phrases ('click here', 'prize reward') while "
            "filtering out rare single-occurrence typos."
        ),
    }

    # 3. Define candidate models
    candidate_models = {
        "Multinomial Naive Bayes": build_pipeline(
            MultinomialNB(alpha=0.2),
            ngram_range=(1, 2),
            max_features=6000,
            min_df=2,
        ),
        "Logistic Regression": build_pipeline(
            LogisticRegression(
                C=3.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
            ngram_range=(1, 2),
            max_features=6000,
            min_df=2,
        ),
    }

    # 4. Fit pipelines on training data ONLY and evaluate on held-out test set
    trained_pipelines: Dict[str, Pipeline] = {}
    evaluations: Dict[str, Dict[str, Any]] = {}

    for model_name, pipeline in candidate_models.items():
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline
        evaluations[model_name] = evaluate_model(
            model=pipeline,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
        )

    # 5. Select the best model
    best_model_name, selection_reason = select_best_model(evaluations)
    best_pipeline = trained_pipelines[best_model_name]
    best_metrics = evaluations[best_model_name]

    # Extract vocabulary size and top features from the winning pipeline
    vocab_size = len(best_pipeline.named_steps["tfidf"].vocabulary_)
    feature_insights = extract_top_tfidf_features(best_pipeline, top_n=20)

    # 6. Save model and metadata
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)

    split_info = {
        "total_samples": len(X),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_spam_count": int(np.sum(y_train == 1)),
        "train_ham_count": int(np.sum(y_train == 0)),
        "test_spam_count": int(np.sum(y_test == 1)),
        "test_ham_count": int(np.sum(y_test == 0)),
        "test_split_ratio": test_size,
        "random_state": random_state,
        "stratified": True,
    }

    metadata: Dict[str, Any] = {
        "project_name": "MAILGUARD AI - Email Spam Detection System",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": best_model_name,
        "selection_reason": selection_reason,
        "model_file": str(model_path.name),
        "vocabulary_size": vocab_size,
        "tfidf_configuration": tfidf_config,
        "dataset_statistics": preprocessing_stats,
        "split_information": split_info,
        "best_model_metrics": best_metrics,
        "model_comparison": evaluations,
        "feature_insights": feature_insights,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata

