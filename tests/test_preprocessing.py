"""
Unit Tests for NLP Text Preprocessing and Signal Extraction (`src/preprocessing.py`).
"""

from src.preprocessing import (
    EmailTextPreprocessor,
    extract_email_signals,
    preprocess_text,
)


def test_lowercase_and_whitespace_normalization() -> None:
    raw = "   HELLO   World \n\t  Meeting   Tomorrow  "
    cleaned = preprocess_text(raw)
    assert cleaned == "hello world meeting tomorrow"


def test_preserve_spam_indicators_and_exclamations() -> None:
    raw = "FREE MONEY!!! Click here now! Congratulations, you won $1000!"
    cleaned = preprocess_text(raw)
    assert "free" in cleaned
    assert "money" in cleaned
    assert "multiexclaimtoken" in cleaned
    assert "exclaimtoken" in cleaned
    assert "currencytoken" in cleaned


def test_url_and_email_normalization() -> None:
    raw = "Visit https://spam-prize.com/claim or contact winner@lottery.org now"
    cleaned = preprocess_text(raw)
    assert "urltoken" in cleaned
    assert "emailtoken" in cleaned
    assert "https" not in cleaned


def test_html_tag_removal() -> None:
    raw = "<html><body><h1>URGENT OFFER!</h1><p>Click <a href='http://deal.com'>here</a> &amp; win</p></body></html>"
    cleaned = preprocess_text(raw)
    assert "html" not in cleaned
    assert "body" not in cleaned
    assert "urgent" in cleaned
    assert "offer" in cleaned
    assert "urltoken" in cleaned


def test_empty_and_none_inputs() -> None:
    assert preprocess_text("") == ""
    assert preprocess_text("     \n\t  ") == ""
    assert preprocess_text(None) == ""


def test_very_short_and_long_messages() -> None:
    short_msg = "Ok"
    assert preprocess_text(short_msg) == "ok"

    long_msg = ("Project update regarding the software architecture review. " * 100).strip()
    cleaned_long = preprocess_text(long_msg)
    assert "project" in cleaned_long
    assert "architecture" in cleaned_long
    assert len(cleaned_long) > 500


def test_extract_email_signals() -> None:
    raw = "URGENT! Claim your FREE prize at http://winner.com! Contact claims@win.org!"
    signals = extract_email_signals(raw)
    assert signals["url_count"] == 1
    assert signals["email_address_count"] == 1
    assert signals["exclamation_count"] == 3
    assert "urgent" in signals["matched_keywords"]
    assert "free" in signals["matched_keywords"]
    assert "prize" in signals["matched_keywords"]


def test_sklearn_transformer_interface() -> None:
    transformer = EmailTextPreprocessor(remove_stopwords=True)
    docs = ["Hello team!", "FREE CASH PRIZE!!!"]
    transformed = transformer.fit_transform(docs)
    assert len(transformed) == 2
    assert "hello" in transformed[0]
    assert "free" in transformed[1]
    assert "multiexclaimtoken" in transformed[1]

