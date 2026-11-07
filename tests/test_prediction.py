"""
Unit Tests for Model Loading, Single Email Prediction, and Bulk CSV Inference (`src/prediction.py`).

Covers all 11 required testing scenarios:
1. Normal legitimate email.
2. Obvious spam email.
3. Empty input.
4. Very short message.
5. Long email content.
6. Email containing URLs.
7. Email containing HTML.
8. Invalid CSV upload.
9. CSV with missing columns.
10. Missing trained model.
11. Model loading failure (corrupted file).
"""

import io
from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import DatasetError, read_csv_safely
from src.prediction import (
    ModelLoadError,
    PredictionInputError,
    load_trained_model,
    predict_bulk_emails,
    predict_single_email,
)


@pytest.fixture(scope="module")
def trained_pipeline_and_meta():
    """Load the actual trained production pipeline from `models/best_model.joblib`."""
    pipeline, metadata = load_trained_model()
    return pipeline, metadata


def test_scenario_1_normal_legitimate_email(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    email = (
        "Hi Sarah, can we reschedule our project review meeting to 3pm tomorrow? "
        "I have finished the slides and will bring the printed report."
    )
    result = predict_single_email(email, pipeline)
    assert result["predicted_label"] == "HAM"
    assert result["predicted_class"] == 0
    assert result["ham_probability"] > 0.5


def test_scenario_2_obvious_spam_email(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    email = (
        "URGENT! WINNER!! You have won a £1000 FREE prize jackpot! "
        "Call 09061701461 or text CLAIM to 81010 now to receive your cash reward!"
    )
    result = predict_single_email(email, pipeline)
    assert result["predicted_label"] == "SPAM"
    assert result["predicted_class"] == 1
    assert result["spam_probability"] > 0.5


def test_scenario_3_empty_input(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    with pytest.raises(PredictionInputError):
        predict_single_email("", pipeline)

    with pytest.raises(PredictionInputError):
        predict_single_email("   \n\t  ", pipeline)

    with pytest.raises(PredictionInputError):
        predict_single_email("...,,,---", pipeline)


def test_scenario_4_very_short_message(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    result = predict_single_email("Ok see you soon", pipeline)
    assert result["predicted_label"] in ("HAM", "SPAM")
    assert result["confidence_percent"] is not None


def test_scenario_5_long_email_content(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    long_email = (
        "Dear Professor, attached is our semester project report on database indexing "
        "and query optimization. We ran benchmarks across multiple workloads and documented "
        "all results in Section 4. Please let us know if any revisions are needed before Friday. "
    ) * 25
    result = predict_single_email(long_email, pipeline)
    assert result["predicted_label"] == "HAM"
    assert result["signals"]["word_count"] > 500


def test_scenario_6_email_containing_urls(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    url_email = (
        "Congratulations! Click here http://www.free-cash-winner.com/claim to claim "
        "your £500 prize voucher right now!"
    )
    result = predict_single_email(url_email, pipeline)
    assert result["signals"]["url_count"] >= 1
    assert "urltoken" in result["cleaned_text"]
    assert result["predicted_label"] == "SPAM"


def test_scenario_7_email_containing_html(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    html_email = (
        "<html><body><h1>FREE ENTRY!</h1><p>Win £1000 cash prize now! "
        "<a href='http://win-prize.co.uk'>Click here</a> to claim!</p></body></html>"
    )
    result = predict_single_email(html_email, pipeline)
    assert "<html>" not in result["cleaned_text"]
    assert result["predicted_label"] == "SPAM"


def test_scenario_8_invalid_csv_upload() -> None:
    empty_stream = io.StringIO("")
    with pytest.raises(DatasetError):
        read_csv_safely(empty_stream)


def test_scenario_9_csv_with_missing_columns(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    df = pd.DataFrame({"wrong_column": ["Hello team", "Free prize claim now"]})
    with pytest.raises(PredictionInputError, match="does not exist"):
        predict_bulk_emails(df, text_column="email_text", pipeline=pipeline)


def test_bulk_prediction_handles_empty_rows_gracefully(trained_pipeline_and_meta) -> None:
    pipeline, _ = trained_pipeline_and_meta
    df = pd.DataFrame(
        {
            "message": [
                "Hi, are we still meeting for coffee at 2 PM?",
                "URGENT! You have won a FREE £1000 prize! Call 08002986030 to claim!",
                "   ",
                None,
            ]
        }
    )
    classified_df, summary = predict_bulk_emails(df, text_column="message", pipeline=pipeline)
    assert summary["total_rows"] == 4
    assert summary["valid_rows"] == 2
    assert summary["empty_rows"] == 2
    assert summary["spam_count"] == 1
    assert summary["ham_count"] == 1
    assert classified_df.loc[0, "predicted_label"] == "HAM"
    assert classified_df.loc[1, "predicted_label"] == "SPAM"
    assert classified_df.loc[2, "predicted_label"] == "EMPTY / INVALID"


def test_scenario_10_missing_trained_model(tmp_path: Path) -> None:
    missing_model_path = tmp_path / "does_not_exist.joblib"
    with pytest.raises(ModelLoadError, match="not found"):
        load_trained_model(model_path=missing_model_path)


def test_scenario_11_model_loading_failure_corrupted_file(tmp_path: Path) -> None:
    corrupted_model = tmp_path / "corrupted_model.joblib"
    corrupted_model.write_bytes(b"NOT_A_VALID_JOBLIB_BINARY_STREAM_12345")
    with pytest.raises(ModelLoadError, match="corrupted"):
        load_trained_model(model_path=corrupted_model)


def test_streamlit_app_launches_without_errors() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception

    # Navigate and verify all 6 application pages render without any runtime exceptions
    pages = [
        "🏠 Home Dashboard",
        "🔍 Email Spam Detection",
        "📂 Bulk Email Classification",
        "📊 Model Analytics",
        "🕒 Prediction History",
        "📘 About Project",
    ]
    for page_name in pages:
        at.sidebar.radio[0].set_value(page_name).run()
        assert not at.exception, f"Exception occurred on page {page_name}: {at.exception}"



