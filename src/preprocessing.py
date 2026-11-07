"""
Natural Language Processing (NLP) and Text Preprocessing Module for MailGuard AI.

Implements text cleaning, normalization, spam-signal preservation, and a
Scikit-learn compatible Transformer (`EmailTextPreprocessor`) so that the
exact same preprocessing logic is encapsulated inside the saved model pipeline.
"""

import html
import re
from typing import Any, Dict, Iterable, List

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError:
    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class TransformerMixin:  # type: ignore[no-redef]
        def fit_transform(self, X: Any, y: Any = None) -> Any:
            return self.fit(X, y).transform(X)

# Regular expressions compiled once for fast execution
HTML_ANCHOR_LINK_PATTERN = re.compile(r"<a\b[^>]*href\s*=\s*[^>]+>", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EMAIL_PATTERN = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
URL_PATTERN = re.compile(
    r"(?:https?://|www\.|wap\.)\S+|(?<![@\w.-])\b[a-zA-Z0-9-]+\.(?:com|net|org|co\.uk|uk|info|biz|ru|cn|io|me|ly|tv)\b(?:/\S*)?",
    re.IGNORECASE,
)
CURRENCY_PATTERN = re.compile(r"(?:[$£€₹]|å£|ì¼|\bpounds?\b|\bdollars?\b)", re.IGNORECASE)
PHONE_OR_SHORTCODE_PATTERN = re.compile(r"\b(?:\+?\d[\d\s-]{7,}\d|\d{5,6})\b")
MULTI_EXCLAIM_PATTERN = re.compile(r"!{2,}")
SINGLE_EXCLAIM_PATTERN = re.compile(r"!")
MULTI_QUESTION_PATTERN = re.compile(r"\?{2,}")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Curated list of lightweight stop words that excludes words carrying negation
# or promotional urgency ("free", "now", "win", "call", "claim", "urgent", etc.)
LIGHT_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
        "for", "with", "about", "into", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "any", "both", "each", "few",
        "more", "most", "other", "some", "such", "own", "same", "so", "than",
        "too", "very", "s", "t", "can", "will", "just", "should", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "am", "it", "its", "itself", "they",
        "them", "their", "theirs", "themselves", "what", "which", "who", "whom",
        "this", "that", "these", "those",
    }
)

# Common suspicious / promotional keywords for educational UI highlighting
SUSPICIOUS_KEYWORDS: List[str] = [
    "free", "winner", "won", "urgent", "claim", "prize", "cash", "bonus",
    "guaranteed", "congratulations", "click here", "limited time", "act now",
    "selected", "reward", "jackpot", "credit", "loan", "unsubscribe",
    "verify your account", "bank account", "password", "lottery", "100%",
    "offer", "discount", "ringtone", "voucher", "million", "billion",
]


def preprocess_text(text: Any, remove_stopwords: bool = True) -> str:
    """
    Clean and normalize a single email string while preserving key spam indicators.

    Operations performed:
    1. HTML entity unescaping and HTML tag removal.
    2. URL normalization -> `urltoken` (preserves presence of links).
    3. Email address normalization -> `emailtoken`.
    4. Currency symbol normalization -> `currencytoken`.
    5. Phone number / SMS shortcode normalization -> `phonetoken`.
    6. Excessive punctuation normalization (`!!!` -> `multiexclaimtoken`, `!` -> `exclaimtoken`).
    7. Lowercase conversion.
    8. Removal of unnecessary punctuation / special characters.
    9. Tokenization and optional lightweight stop-word filtering.
    10. Collapse of repeated whitespace.

    Parameters
    ----------
    text : Any
        Raw email message text.
    remove_stopwords : bool, default=True
        Whether to filter out common non-informative stop words.

    Returns
    -------
    str
        Normalized string ready for TF-IDF vectorization.
    """
    if text is None or not isinstance(text, str):
        text = "" if text is None else str(text)

    cleaned = text.strip()
    if not cleaned:
        return ""

    # 1. Decode HTML entities (&amp;, &nbsp;, etc.), preserve <a href="..."> links, and strip HTML tags
    cleaned = html.unescape(cleaned)
    cleaned = HTML_ANCHOR_LINK_PATTERN.sub(" urltoken ", cleaned)
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)

    # 2. Normalize Email addresses first, then URLs into semantic tokens
    cleaned = EMAIL_PATTERN.sub(" emailtoken ", cleaned)
    cleaned = URL_PATTERN.sub(" urltoken ", cleaned)

    # 3. Normalize currency symbols and phone/shortcode numbers
    cleaned = CURRENCY_PATTERN.sub(" currencytoken ", cleaned)
    cleaned = PHONE_OR_SHORTCODE_PATTERN.sub(" phonetoken ", cleaned)

    # 4. Preserve excessive exclamation/question marks as tokens before stripping punctuation
    cleaned = MULTI_EXCLAIM_PATTERN.sub(" multiexclaimtoken ", cleaned)
    cleaned = SINGLE_EXCLAIM_PATTERN.sub(" exclaimtoken ", cleaned)
    cleaned = MULTI_QUESTION_PATTERN.sub(" multiquestiontoken ", cleaned)

    # 5. Convert to lowercase
    cleaned = cleaned.lower()

    # 6. Remove remaining non-alphanumeric punctuation
    cleaned = NON_ALPHANUMERIC_PATTERN.sub(" ", cleaned)

    # 7. Collapse repeated whitespace and optionally filter stop words
    tokens = WHITESPACE_PATTERN.split(cleaned.strip())
    if remove_stopwords:
        filtered_tokens = [tok for tok in tokens if tok and tok not in LIGHT_STOP_WORDS]
        # Fallback to unfiltered tokens if stop-word removal emptied a short phrase
        if filtered_tokens:
            tokens = filtered_tokens

    return " ".join(tokens)


def extract_email_signals(text: str) -> Dict[str, Any]:
    """
    Extract human-readable structural statistics and suspicious keyword matches
    from an email message for display in the UI.

    Note: These indicators are provided for educational inspection; the actual
    classification decision is made by the trained TF-IDF + ML pipeline.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0,
            "word_count": 0,
            "url_count": 0,
            "email_address_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0.0,
            "digit_count": 0,
            "matched_keywords": [],
        }

    words = text.split()
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]

    lower_text = text.lower()
    matched_keywords = [
        kw for kw in SUSPICIOUS_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", lower_text)
    ]

    return {
        "char_count": len(text),
        "word_count": len(words),
        "url_count": len(URL_PATTERN.findall(text)),
        "email_address_count": len(EMAIL_PATTERN.findall(text)),
        "exclamation_count": text.count("!"),
        "uppercase_ratio": round(len(upper_chars) / len(alpha_chars), 3) if alpha_chars else 0.0,
        "digit_count": sum(c.isdigit() for c in text),
        "matched_keywords": matched_keywords,
    }


class EmailTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that applies `preprocess_text`
    to an iterable of raw email strings inside a `Pipeline`.
    """

    def __init__(self, remove_stopwords: bool = True) -> None:
        self.remove_stopwords = remove_stopwords

    def fit(self, X: Iterable[Any], y: Any = None) -> "EmailTextPreprocessor":
        """Stateless transformer: returns self without learning parameters."""
        return self

    def transform(self, X: Iterable[Any]) -> List[str]:
        """Apply `preprocess_text` to every document in X."""
        return [
            preprocess_text(doc, remove_stopwords=self.remove_stopwords)
            for doc in X
        ]
