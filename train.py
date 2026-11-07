"""
Standalone CLI Training Script for MAILGUARD AI.

Usage:
    python train.py

Loads `data/emails.csv`, validates and cleans the dataset, splits 80:20 with
stratification prior to TF-IDF vectorization, trains Multinomial Naive Bayes
and Logistic Regression pipelines, compares evaluation metrics on the test set,
and saves the best pipeline (`models/best_model.joblib`) and metadata
(`models/model_metadata.json`).
"""

import argparse
import sys
from pathlib import Path

from src.data_loader import DatasetError
from src.training import train_and_save_models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Email Spam Detection Machine Learning models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional path to custom CSV dataset (defaults to data/emails.csv).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proportion of dataset reserved for testing (default: 0.20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible train/test splitting (default: 42).",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("  MAILGUARD AI — EMAIL SPAM DETECTION MODEL TRAINING PIPELINE")
    print("=" * 76)

    try:
        metadata = train_and_save_models(
            dataset_path=Path(args.dataset) if args.dataset else None,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except DatasetError as exc:
        print(f"\n[DATASET ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[TRAINING ERROR] {exc}", file=sys.stderr)
        return 1

    ds = metadata["dataset_statistics"]
    split = metadata["split_information"]
    comp = metadata["model_comparison"]
    best_name = metadata["selected_model"]

    print("\n1. DATASET PREPROCESSING SUMMARY")
    print("-" * 76)
    print(f"  Dataset Source          : {ds['dataset_path']}")
    print(f"  Initial Dimensions      : {ds['initial_rows']:,} rows x {ds['initial_columns']} columns")
    print(f"  Detected Text Column    : '{ds['detected_text_column']}'")
    print(f"  Detected Label Column   : '{ds['detected_label_column']}'")
    print(f"  Missing Values Removed  : {ds['total_missing_values']}")
    print(f"  Empty Messages Removed  : {ds['empty_messages_removed']}")
    print(f"  Invalid Labels Removed  : {ds['invalid_labels_removed']}")
    print(f"  Duplicates Removed      : {ds['duplicate_records_removed']:,}")
    print(f"  Final Clean Dataset     : {ds['final_rows']:,} emails")
    print(
        f"  Class Distribution      : HAM = {ds['ham_count']:,} ({ds['ham_percentage']}%) | "
        f"SPAM = {ds['spam_count']:,} ({ds['spam_percentage']}%)"
    )

    print("\n2. TRAIN / TEST SPLIT & TF-IDF CONFIGURATION")
    print("-" * 76)
    print(f"  Training Set (80%)      : {split['train_size']:,} emails "
          f"(HAM: {split['train_ham_count']:,}, SPAM: {split['train_spam_count']:,})")
    print(f"  Testing Set (20%)       : {split['test_size']:,} emails "
          f"(HAM: {split['test_ham_count']:,}, SPAM: {split['test_spam_count']:,})")
    print(f"  Learned TF-IDF Vocab    : {metadata['vocabulary_size']:,} terms (unigrams & bigrams)")

    print("\n3. MODEL COMPARISON ON HELD-OUT TEST SET")
    print("-" * 76)
    header = f"  {'Model Name':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}"
    print(header)
    print("  " + "-" * 72)
    for name, m in comp.items():
        marker = " *" if name == best_name else "  "
        print(
            f"{marker}{name:<28} "
            f"{m['accuracy'] * 100:>9.2f}% "
            f"{m['precision'] * 100:>9.2f}% "
            f"{m['recall'] * 100:>9.2f}% "
            f"{m['f1_score'] * 100:>9.2f}%"
        )

    print("\n4. SELECTED PRODUCTION MODEL")
    print("-" * 76)
    print(f"  Winner                  : {best_name}")
    print(f"  Selection Reason        : {metadata['selection_reason']}")
    print("\n  Classification Report (Best Model):")
    for line in metadata["best_model_metrics"]["classification_report_text"].splitlines():
        print(f"    {line}")

    print("\n5. SAVED ARTIFACTS")
    print("-" * 76)
    print("  Model Pipeline          : models/best_model.joblib")
    print("  Model Metadata          : models/model_metadata.json")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())

