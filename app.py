"""
MAILGUARD AI — Streamlit Web Application for Email Spam Detection.

Run from the project root:
    streamlit run app.py
"""

from datetime import datetime
import io
import json
from typing import Any, Callable, Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import streamlit as st
    from src.data_loader import (
        DatasetError,
        TEXT_COLUMN_CANDIDATES,
        read_csv_safely,
    )
    from src.visualization import (
        ACCENT_BLUE,
        HAM_COLOR,
        SPAM_COLOR,
        create_class_distribution_chart,
        create_confusion_matrix_figure,
        create_model_comparison_chart,
        create_probability_bar_chart,
        create_top_features_chart,
    )
except ImportError:
    plt = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

from src.prediction import (
    ModelLoadError,
    PredictionInputError,
    load_trained_model,
    predict_bulk_emails,
    predict_single_email,
)
from src.web_wsgi import wsgi_app


def app(
    environ: Dict[str, Any],
    start_response: Callable[[str, List[Tuple[str, str]]], Any],
) -> Iterable[bytes]:
    """
    Top-level PEP 3333 WSGI application entrypoint.
    Used by Vercel (@vercel/python), Gunicorn, uWSGI, and other WSGI servers.
    """
    return wsgi_app(environ, start_response)


# Alias for WSGI servers that look for `application`
application = app

# Maximum bulk CSV file size in megabytes
MAX_UPLOAD_SIZE_MB = 25

# Curated sample emails for quick one-click testing
SAMPLE_EMAILS: Dict[str, str] = {
    "Promotional Prize Spam": (
        "URGENT! Congratulations, you have been selected to receive a £1000 CASH prize reward! "
        "To claim your guaranteed free bonus now, click here: http://www.prize-claim-win.com "
        "or call 09061701461 within 12 hours! Valid today only!!!"
    ),
    "Phishing / Account Alert Spam": (
        "FINAL NOTICE: Your online bank account has been temporarily suspended due to unusual "
        "activity. Verify your account password and credit card details immediately at "
        "http://secure-login-update.net/verify to avoid permanent closure. Free security update!"
    ),
    "Legitimate Work Email (Ham)": (
        "Hi team, just a quick reminder that our sprint planning meeting is scheduled for "
        "tomorrow at 10:30 AM in Conference Room B. Please review the project backlog and "
        "let me know if we need to adjust any engineering tasks before the call. Thanks, Alex"
    ),
    "Legitimate Personal Message (Ham)": (
        "Hey! Are we still meeting for lunch around 1:00 PM after the lecture? "
        "I finished my database assignment early so I can bring the notes you asked for. "
        "See you at the campus cafe!"
    ),
    "HTML & Link Marketing Spam": (
        "<html><body><h2>FREE MEMBERSHIP OFFER!!!</h2><p>Congratulations! You won a 1 week "
        "FREE VIP voucher worth $500! <a href='https://free-voucher-deals.biz'>Click here now!</a> "
        "Reply CLAIM to 81010. T&amp;Cs apply.</p></body></html>"
    ),
}


def inject_custom_css() -> None:
    """Apply dark-navy, slate, and purple/blue cybersecurity theme styling."""
    st.markdown(
        """
        <style>
        /* Main application dark navy theme */
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
        }
        [data-testid="stSidebar"] {
            background-color: #1E293B;
            border-right: 1px solid #334155;
        }
        /* Header banner */
        .hero-banner {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 60%, #1E1B4B 100%);
            border: 1px solid #334155;
            border-left: 5px solid #6366F1;
            border-radius: 12px;
            padding: 1.4rem 1.8rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            color: #F8FAFC;
            margin: 0;
        }
        .hero-tagline {
            font-size: 1.02rem;
            color: #94A3B8;
            margin-top: 0.35rem;
            margin-bottom: 0;
        }
        /* Metric & Content Cards */
        .metric-card {
            background-color: #1E293B;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #F8FAFC;
            margin: 0;
        }
        .metric-sub {
            font-size: 0.8rem;
            color: #6366F1;
            margin-top: 0.25rem;
        }
        /* Workflow step pill */
        .workflow-box {
            background-color: #1E293B;
            border: 1px solid #4F46E5;
            border-radius: 10px;
            padding: 0.85rem 0.6rem;
            text-align: center;
            font-weight: 600;
            color: #E2E8F0;
        }
        /* Prediction banners */
        .spam-result-box {
            background: rgba(239, 68, 68, 0.14);
            border: 2px solid #EF4444;
            border-radius: 12px;
            padding: 1.4rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .ham-result-box {
            background: rgba(16, 185, 129, 0.14);
            border: 2px solid #10B981;
            border-radius: 12px;
            padding: 1.4rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .keyword-badge {
            display: inline-block;
            background-color: rgba(239, 68, 68, 0.22);
            border: 1px solid #EF4444;
            color: #FCA5A5;
            border-radius: 6px;
            padding: 0.2rem 0.55rem;
            margin: 0.18rem;
            font-size: 0.84rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_model_and_metadata():
    """Load the trained model pipeline and metadata JSON."""
    return load_trained_model()


get_cached_model_and_metadata = (
    st.cache_resource(show_spinner="Loading trained MailGuard AI model pipeline...")(_load_model_and_metadata)
    if st is not None
    else _load_model_and_metadata
)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "email_input_text" not in st.session_state:
        st.session_state["email_input_text"] = ""
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None


def render_hero_header() -> None:
    """Render the top MailGuard AI cybersecurity banner."""
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🛡️ MAILGUARD AI</div>
            <div class="hero-tagline">
                Intelligent Email Spam Detection Powered by Machine Learning
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtitle: str = "") -> None:
    """Render a styled metric card."""
    sub_html = f'<div class="metric-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: HOME DASHBOARD
# ==============================================================================
def render_home_dashboard(metadata: Dict[str, Any]) -> None:
    st.subheader("🏠 Executive Security & Model Overview")
    st.write(
        "**MAILGUARD AI** is a Supervised Machine Learning web application that analyzes "
        "email content using Natural Language Processing (NLP) and Term Frequency–Inverse "
        "Document Frequency (**TF-IDF**) feature extraction to classify messages as "
        "**SPAM** or **HAM (Legitimate)** in real time."
    )

    ds = metadata.get("dataset_statistics", {})
    best_metrics = metadata.get("best_model_metrics", {})
    selected_model = metadata.get("selected_model", "Unknown")

    total_emails = ds.get("final_rows", 0)
    spam_emails = ds.get("spam_count", 0)
    ham_emails = ds.get("ham_count", 0)
    accuracy_pct = f"{best_metrics.get('accuracy', 0.0) * 100:.2f}%"
    recall_pct = f"{best_metrics.get('recall', 0.0) * 100:.2f}%"
    precision_pct = f"{best_metrics.get('precision', 0.0) * 100:.2f}%"

    # Row 1: Dataset & Model KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            "Total Cleaned Emails",
            f"{total_emails:,}",
            f"Raw rows: {ds.get('initial_rows', 0):,} ({ds.get('duplicate_records_removed', 0):,} duplicates removed)",
        )
    with col2:
        render_metric_card(
            "Total Spam Emails (1)",
            f"{spam_emails:,}",
            f"{ds.get('spam_percentage', 0):.2f}% of dataset",
        )
    with col3:
        render_metric_card(
            "Total Legitimate Emails (0)",
            f"{ham_emails:,}",
            f"{ds.get('ham_percentage', 0):.2f}% of dataset",
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        render_metric_card(
            "Selected ML Model",
            selected_model,
            f"TF-IDF Vocab: {metadata.get('vocabulary_size', 0):,} terms",
        )
    with col5:
        render_metric_card(
            "Test Set Accuracy",
            accuracy_pct,
            f"Spam Precision: {precision_pct}",
        )
    with col6:
        render_metric_card(
            "Spam Detection Recall",
            recall_pct,
            f"Spam F1-Score: {best_metrics.get('f1_score', 0.0) * 100:.2f}%",
        )

    st.markdown("---")

    # System Workflow Diagram
    st.markdown("### ⚙️ End-to-End Classification Workflow")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        st.markdown(
            '<div class="workflow-box">1️⃣ Email Input<br><small>Raw Text / CSV</small></div>',
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            '<div class="workflow-box">2️⃣ Text Processing<br><small>Clean, Normalize & Tokenize</small></div>',
            unsafe_allow_html=True,
        )
    with w3:
        st.markdown(
            '<div class="workflow-box">3️⃣ TF-IDF Vectorizer<br><small>Unigrams & Bigrams (1,2)</small></div>',
            unsafe_allow_html=True,
        )
    with w4:
        st.markdown(
            f'<div class="workflow-box">4️⃣ ML Classifier<br><small>{selected_model}</small></div>',
            unsafe_allow_html=True,
        )
    with w5:
        st.markdown(
            '<div class="workflow-box">5️⃣ Prediction<br><small>SPAM / HAM + Probability</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Distribution Chart & Preprocessing Audit Log
    left_col, right_col = st.columns([1.05, 1])
    with left_col:
        fig_dist = create_class_distribution_chart(spam_count=spam_emails, ham_count=ham_emails)
        st.plotly_chart(fig_dist, width="stretch")

    with right_col:
        st.markdown("#### 🧹 Dataset Preprocessing & Audit Statistics")
        audit_rows = [
            ("Dataset File Path", ds.get("dataset_path", "data/emails.csv")),
            ("Initial Dataset Shape", f"{ds.get('initial_rows', 0):,} rows × {ds.get('initial_columns', 0)} columns"),
            ("Auto-Detected Text Column", f"`{ds.get('detected_text_column', 'v2')}`"),
            ("Auto-Detected Label Column", f"`{ds.get('detected_label_column', 'v1')}`"),
            ("Missing Values Removed", str(ds.get("total_missing_values", 0))),
            ("Empty Messages Removed", str(ds.get("empty_messages_removed", 0))),
            ("Invalid Labels Removed", str(ds.get("invalid_labels_removed", 0))),
            ("Duplicate Emails Removed", f"{ds.get('duplicate_records_removed', 0):,}"),
            ("Final Cleaned Dataset Size", f"{ds.get('final_rows', 0):,} emails"),
            ("Avg Spam vs Ham Length", f"{ds.get('avg_spam_char_length', 0)} chars vs {ds.get('avg_ham_char_length', 0)} chars"),
        ]
        audit_df = pd.DataFrame(audit_rows, columns=["Preprocessing Step", "Recorded Value"])
        st.dataframe(audit_df, width="stretch", hide_index=True)


# ==============================================================================
# PAGE 2: EMAIL SPAM DETECTION
# ==============================================================================
def render_single_detection_page(pipeline: Any, metadata: Dict[str, Any]) -> None:
    st.subheader("🔍 Real-Time Email Spam Detection")
    st.write(
        "Paste an email message below or load one of the sample messages to inspect its "
        "classification, class probabilities, and structural indicators."
    )

    # Sample email buttons
    st.markdown("**Quick Test with Sample Emails:**")
    sample_cols = st.columns(len(SAMPLE_EMAILS))
    for idx, (sample_label, sample_body) in enumerate(SAMPLE_EMAILS.items()):
        with sample_cols[idx]:
            if st.button(sample_label, key=f"sample_btn_{idx}", width="stretch"):
                st.session_state["email_input_text"] = sample_body
                st.session_state["last_prediction"] = None

    email_text = st.text_area(
        "Enter Email Content to Analyze:",
        key="email_input_text",
        height=190,
        placeholder="Paste the email subject and body here (e.g., 'Congratulations! You have won a $1,000 prize...').",
    )

    btn_col1, btn_col2, _ = st.columns([1.2, 1, 3])
    with btn_col1:
        predict_clicked = st.button("🛡️ Analyze & Predict Email", type="primary", width="stretch")
    with btn_col2:
        if st.button("🗑️ Clear Input", width="stretch"):
            st.session_state["email_input_text"] = ""
            st.session_state["last_prediction"] = None
            st.rerun()

    if predict_clicked:
        try:
            result = predict_single_email(email_text, pipeline)
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["model_used"] = metadata.get("selected_model", "Trained Pipeline")
            st.session_state["last_prediction"] = result

            # Save to session prediction history
            st.session_state["prediction_history"].insert(
                0,
                {
                    "timestamp": result["timestamp"],
                    "predicted_label": result["predicted_label"],
                    "spam_probability": (
                        f"{result['spam_probability'] * 100:.2f}%"
                        if result["spam_probability"] is not None
                        else "N/A"
                    ),
                    "confidence": (
                        f"{result['confidence_percent']:.2f}%"
                        if result["confidence_percent"] is not None
                        else "N/A"
                    ),
                    "email_preview": (
                        result["raw_text"][:120] + "..."
                        if len(result["raw_text"]) > 120
                        else result["raw_text"]
                    ),
                    "full_text": result["raw_text"],
                },
            )
        except PredictionInputError as exc:
            st.session_state["last_prediction"] = None
            st.warning(f"⚠️ **Input Validation Notice:** {exc}")
        except Exception as exc:
            st.session_state["last_prediction"] = None
            st.error(f"❌ **Prediction Error:** An unexpected error occurred while classifying the message: {exc}")

    # Display result if available
    pred = st.session_state.get("last_prediction")
    if pred:
        is_spam = pred["predicted_class"] == 1
        spam_prob = pred["spam_probability"]
        ham_prob = pred["ham_probability"]
        confidence = pred["confidence_percent"]

        if is_spam:
            st.markdown(
                f"""
                <div class="spam-result-box">
                    <h3 style="color: #FCA5A5; margin: 0;">🚨 CLASSIFICATION RESULT: SPAM DETECTED</h3>
                    <p style="color: #F8FAFC; margin-top: 0.5rem; margin-bottom: 0;">
                        The Machine Learning model classified this email content as <b>suspicious / unsolicited spam</b>.
                        Exercise caution with any embedded links, phone numbers, or requests for sensitive information.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.error("⚠️ **Security Alert:** High similarity to known promotional, phishing, or spam patterns.")
        else:
            st.markdown(
                f"""
                <div class="ham-result-box">
                    <h3 style="color: #6EE7B7; margin: 0;">✅ CLASSIFICATION RESULT: HAM (LEGITIMATE EMAIL)</h3>
                    <p style="color: #F8FAFC; margin-top: 0.5rem; margin-bottom: 0;">
                        The Machine Learning model classified this email as <b>legitimate (normal communication)</b>.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success("✅ **Status:** No dominant spam patterns detected by the model.")

        # Probability metrics & chart
        if spam_prob is not None and ham_prob is not None:
            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("Predicted Class", pred["predicted_label"], f"Model: {pred['model_used']}")
            with m2:
                render_metric_card("Spam Probability", f"{spam_prob * 100:.2f}%", "Likelihood of class 1 (SPAM)")
            with m3:
                render_metric_card(
                    "Model Confidence",
                    f"{confidence:.2f}%",
                    f"Legitimate (HAM) Probability: {ham_prob * 100:.2f}%",
                )

            st.plotly_chart(
                create_probability_bar_chart(spam_prob=spam_prob, ham_prob=ham_prob),
                width="stretch",
            )

        # Structural Statistics & Suspicious Keyword Highlighting
        signals = pred["signals"]
        with st.expander("🔎 View NLP Preprocessing & Email Signal Breakdown", expanded=True):
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Word Count", signals["word_count"])
            s2.metric("Character Length", signals["char_count"])
            s3.metric("URLs Detected", signals["url_count"])
            s4.metric("Exclamation Marks (!)", signals["exclamation_count"])
            s5.metric("Uppercase Letter Ratio", f"{signals['uppercase_ratio'] * 100:.1f}%")

            if signals["matched_keywords"]:
                badges_html = "".join(
                    f'<span class="keyword-badge">{kw}</span>' for kw in signals["matched_keywords"]
                )
                st.markdown(
                    f"**Suspicious / Promotional Keywords Present:**<br>{badges_html}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Note: Highlighted keywords are shown for human inspection. "
                    "Keywords alone do not determine the model's prediction; the classification "
                    "is computed from the full TF-IDF unigram/bigram vector across all words."
                )
            else:
                st.caption("No common promotional trigger keywords found in the raw text.")

            st.markdown("**Normalized Text Passed to TF-IDF Vectorizer:**")
            st.code(pred["cleaned_text"], language="text")

        # Downloadable Prediction Report
        report_payload = json.dumps(pred, indent=2)
        st.download_button(
            label="📥 Download Prediction Report (JSON)",
            data=report_payload,
            file_name=f"mailguard_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    st.markdown("---")
    st.info(
        "ℹ️ **Important Disclaimer:** This application provides a Machine Learning-based "
        "classification and does not guarantee that an email is safe or malicious. "
        "Model confidence reflects statistical probability learned from training data, "
        "not an absolute security guarantee."
    )


# ==============================================================================
# PAGE 3: BULK EMAIL CLASSIFICATION
# ==============================================================================
def render_bulk_classification_page(pipeline: Any) -> None:
    st.subheader("📂 Bulk Email Classification via CSV Upload")
    st.write(
        "Upload a CSV file containing multiple email messages to classify an entire batch at once, "
        "inspect spam vs. legitimate statistics, and download the annotated CSV report."
    )

    # Provide sample CSV download for convenient testing
    sample_csv_df = pd.DataFrame(
        {
            "email_id": [101, 102, 103, 104, 105],
            "message": list(SAMPLE_EMAILS.values()),
        }
    )
    st.download_button(
        label="📄 Download Sample Unlabeled CSV for Testing",
        data=sample_csv_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_bulk_emails.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        f"Upload CSV File (Max {MAX_UPLOAD_SIZE_MB} MB):",
        type=["csv"],
        help="Upload a valid .csv file containing at least one text column with email messages.",
    )

    if uploaded_file is not None:
        # Check file size limit
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_SIZE_MB:
            st.error(
                f"❌ **File Too Large:** Uploaded file is {file_size_mb:.2f} MB, which exceeds "
                f"the {MAX_UPLOAD_SIZE_MB} MB safety limit."
            )
            return

        if uploaded_file.size == 0:
            st.error("❌ **Empty File:** The uploaded CSV file is empty (0 bytes).")
            return

        try:
            raw_df = read_csv_safely(uploaded_file)
        except DatasetError as exc:
            st.error(f"❌ **Invalid CSV File:** {exc}")
            return
        except Exception as exc:
            st.error(f"❌ **Error Reading CSV:** {exc}")
            return

        available_columns = list(raw_df.columns)
        if not available_columns:
            st.error("❌ **Missing Columns:** No valid columns were detected in the uploaded CSV.")
            return

        # Automatically suggest the best candidate text column
        default_col_index = 0
        for idx, col in enumerate(available_columns):
            if str(col).strip().lower() in TEXT_COLUMN_CANDIDATES:
                default_col_index = idx
                break

        st.markdown("#### 1️⃣ Configure Email Content Column")
        selected_text_col = st.selectbox(
            "Select the column containing email text:",
            options=available_columns,
            index=default_col_index,
        )

        if st.button("🚀 Run Bulk Classification", type="primary"):
            with st.spinner(f"Classifying {len(raw_df):,} records..."):
                try:
                    classified_df, summary = predict_bulk_emails(
                        df=raw_df,
                        text_column=selected_text_col,
                        pipeline=pipeline,
                    )
                except PredictionInputError as exc:
                    st.error(f"❌ **Validation Error:** {exc}")
                    return
                except Exception as exc:
                    st.error(f"❌ **Bulk Classification Error:** {exc}")
                    return

            st.success(
                f"✅ Successfully processed **{summary['valid_rows']:,}** valid messages "
                f"out of **{summary['total_rows']:,}** total rows."
            )

            if summary["empty_rows"] > 0:
                st.warning(
                    f"⚠️ **{summary['empty_rows']}** row(s) were empty or contained no valid words "
                    "and were marked as `EMPTY / INVALID`."
                )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Total Rows Uploaded", f"{summary['total_rows']:,}", f"Valid: {summary['valid_rows']:,}")
            with c2:
                render_metric_card("Predicted SPAM", f"{summary['spam_count']:,}", f"{summary['spam_percentage']:.2f}% of valid emails")
            with c3:
                render_metric_card("Predicted HAM", f"{summary['ham_count']:,}", f"{summary['ham_percentage']:.2f}% of valid emails")
            with c4:
                render_metric_card("Empty / Skipped Rows", f"{summary['empty_rows']:,}", "Gracefully handled")

            st.markdown("#### 2️⃣ Classified Dataset Results")
            st.dataframe(classified_df, width="stretch", height=380)

            csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Classified Results (CSV)",
                data=csv_bytes,
                file_name=f"mailguard_classified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary",
            )


# ==============================================================================
# PAGE 4: MODEL ANALYTICS
# ==============================================================================
def render_model_analytics_page(metadata: Dict[str, Any]) -> None:
    st.subheader("📊 Machine Learning Model Evaluation & Analytics")
    st.write(
        "All metrics displayed below are computed strictly on the **held-out 20% test dataset** "
        "(unseen during training and TF-IDF vocabulary fitting) to prevent data leakage."
    )

    selected_model = metadata.get("selected_model", "N/A")
    selection_reason = metadata.get("selection_reason", "")
    best_metrics = metadata.get("best_model_metrics", {})
    comparison = metadata.get("model_comparison", {})
    split_info = metadata.get("split_information", {})

    st.info(f"🏆 **Selected Production Model: {selected_model}**\n\n{selection_reason}")

    # Top-level metrics for selected model
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Test Accuracy", f"{best_metrics.get('accuracy', 0) * 100:.2f}%", "Overall correct predictions")
    with m2:
        render_metric_card("Spam Precision", f"{best_metrics.get('precision', 0) * 100:.2f}%", "Low False Positive rate")
    with m3:
        render_metric_card("Spam Recall", f"{best_metrics.get('recall', 0) * 100:.2f}%", "Proportion of actual spam caught")
    with m4:
        render_metric_card("Spam F1-Score", f"{best_metrics.get('f1_score', 0) * 100:.2f}%", "Harmonic mean of Precision & Recall")

    # Train/Test Split Information
    s1, s2, s3 = st.columns(3)
    with s1:
        render_metric_card(
            "Training Set Size (80%)",
            f"{split_info.get('train_size', 0):,}",
            f"HAM: {split_info.get('train_ham_count', 0):,} | SPAM: {split_info.get('train_spam_count', 0):,}",
        )
    with s2:
        render_metric_card(
            "Testing Set Size (20%)",
            f"{split_info.get('test_size', 0):,}",
            f"HAM: {split_info.get('test_ham_count', 0):,} | SPAM: {split_info.get('test_spam_count', 0):,}",
        )
    with s3:
        roc_val = best_metrics.get("roc_auc")
        render_metric_card(
            "ROC-AUC Score",
            f"{roc_val * 100:.2f}%" if roc_val is not None else "N/A",
            "Separability across thresholds",
        )

    st.markdown("---")

    # Model Comparison Table & Bar Chart
    st.markdown("### ⚖️ Multinomial Naive Bayes vs. Logistic Regression")
    if comparison:
        comp_rows = []
        for name, m in comparison.items():
            comp_rows.append(
                {
                    "Model": f"⭐ {name}" if name == selected_model else name,
                    "Accuracy (%)": round(m["accuracy"] * 100, 2),
                    "Spam Precision (%)": round(m["precision"] * 100, 2),
                    "Spam Recall (%)": round(m["recall"] * 100, 2),
                    "Spam F1-Score (%)": round(m["f1_score"] * 100, 2),
                    "Macro F1 (%)": round(m["macro_f1"] * 100, 2),
                    "False Positives (FP)": m["false_positives"],
                    "False Negatives (FN)": m["false_negatives"],
                }
            )
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, width="stretch", hide_index=True)

        st.plotly_chart(
            create_model_comparison_chart(comparison),
            width="stretch",
        )

    st.markdown("---")

    # Confusion Matrix Heatmaps side-by-side
    st.markdown("### 🧩 Confusion Matrix Analysis (Test Set)")
    cm_cols = st.columns(len(comparison)) if comparison else []
    for idx, (model_name, m_data) in enumerate(comparison.items()):
        with cm_cols[idx]:
            fig_cm = create_confusion_matrix_figure(m_data["confusion_matrix"], model_name)
            st.pyplot(fig_cm, width="stretch")
            plt.close(fig_cm)
            st.caption(
                f"**{model_name}:** True Negatives (Correct Ham) = **{m_data['true_negatives']}**, "
                f"False Positives (Legitimate marked Spam) = **{m_data['false_positives']}**, "
                f"False Negatives (Missed Spam) = **{m_data['false_negatives']}**, "
                f"True Positives (Caught Spam) = **{m_data['true_positives']}**."
            )

    # Educational Explanation of Precision, Recall, FP, FN
    with st.expander("📖 Understanding Precision, Recall, False Positives & False Negatives", expanded=True):
        st.markdown(
            """
            - **Why Accuracy Alone Is Not Enough:** Because approximately **87%** of real-world messages in the dataset are legitimate (**HAM**) and only **13%** are **SPAM**, a naive model that predicts "HAM" for every single email would still achieve 87% accuracy while catching **0% of spam**! Therefore, we prioritize **Spam Precision**, **Spam Recall**, and **Spam F1-Score**.
            - **Spam Precision ($\\frac{TP}{TP + FP}$):** Out of all emails the model flagged as SPAM, what percentage was actually SPAM? High precision prevents **False Positives**.
            - **Spam Recall ($\\frac{TP}{TP + FN}$):** Out of all real SPAM emails in the test set, what percentage did the model successfully catch? High recall prevents **False Negatives**.
            - **False Positive (FP):** Occurs when a **legitimate email is incorrectly classified as SPAM**. In email systems, this is critical because users might miss important work or personal emails placed in the spam folder.
            - **False Negative (FN):** Occurs when a **spam email is incorrectly classified as legitimate (HAM)**, allowing unwanted or phishing mail into the user's inbox.
            """
        )

    st.markdown("---")

    # Top TF-IDF Features Visualization
    st.markdown("### 🔤 Top Learned TF-IDF Indicators")
    st.write(
        "The chart below displays the highest-weighted unigrams and bigrams learned by the "
        f"**{selected_model}** pipeline from the training set."
    )
    feature_insights = metadata.get("feature_insights", {})
    top_spam = feature_insights.get("top_spam_features", [])
    top_ham = feature_insights.get("top_ham_features", [])

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.plotly_chart(
            create_top_features_chart(
                top_spam,
                title="Top 15 Spam-Indicating Terms / Tokens",
                bar_color=SPAM_COLOR,
            ),
            width="stretch",
        )
    with f_col2:
        st.plotly_chart(
            create_top_features_chart(
                top_ham,
                title="Top 15 Legitimate (Ham) Terms / Tokens",
                bar_color=HAM_COLOR,
            ),
            width="stretch",
        )

    # Detailed Classification Report
    st.markdown("### 📋 Detailed Scikit-Learn Classification Report")
    st.code(best_metrics.get("classification_report_text", "Report unavailable"), language="text")


# ==============================================================================
# PAGE 5: PREDICTION HISTORY
# ==============================================================================
def render_prediction_history_page() -> None:
    st.subheader("🕒 Session Prediction History")
    st.info(
        "ℹ️ **Session Storage Notice:** Prediction history is maintained in Streamlit session state "
        "for privacy and simplicity. Your history automatically resets when this browser session ends."
    )

    history: List[Dict[str, Any]] = st.session_state.get("prediction_history", [])

    if not history:
        st.write("No emails have been analyzed in this session yet. Go to **Email Spam Detection** to test a message!")
        return

    top_col1, top_col2, _ = st.columns([1.2, 1.5, 3])
    with top_col1:
        if st.button("🗑️ Clear Prediction History", width="stretch"):
            st.session_state["prediction_history"] = []
            st.session_state["last_prediction"] = None
            st.rerun()

    hist_df = pd.DataFrame(history)
    with top_col2:
        st.download_button(
            label="📥 Download History (CSV)",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name=f"mailguard_session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )

    spam_in_session = sum(1 for item in history if item["predicted_label"] == "SPAM")
    ham_in_session = len(history) - spam_in_session

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Total Session Checks", str(len(history)))
    with c2:
        render_metric_card("Flagged as SPAM", str(spam_in_session))
    with c3:
        render_metric_card("Classified as HAM", str(ham_in_session))

    display_cols = ["timestamp", "predicted_label", "spam_probability", "confidence", "email_preview"]
    st.dataframe(
        hist_df[display_cols].rename(
            columns={
                "timestamp": "Timestamp",
                "predicted_label": "Predicted Class",
                "spam_probability": "Spam Probability",
                "confidence": "Model Confidence",
                "email_preview": "Email Content Preview",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# ==============================================================================
# PAGE 6: ABOUT PROJECT
# ==============================================================================
def render_about_project_page(metadata: Dict[str, Any]) -> None:
    st.subheader("📘 About MAILGUARD AI")

    st.markdown(
        """
        ### 1. Project Objective & Problem Statement
        Email and messaging platforms remain the primary vector for unsolicited commercial spam, phishing lures,
        and fraudulent prize scams. Manual rule-based filters quickly become outdated as spammers alter spelling
        and formatting.

        **Objective:** Design and implement a complete, modular Supervised Machine Learning system capable of
        automatically learning linguistic and structural spam patterns from labeled email datasets and classifying
        unseen messages with high precision and recall.

        ---

        ### 2. Technology Stack
        - **Programming Language:** Python 3.10+
        - **Data Processing & Manipulation:** Pandas, NumPy
        - **Machine Learning & Pipelines:** Scikit-learn (`Pipeline`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`)
        - **Natural Language Processing (NLP):** Custom Regex Normalization (`EmailTextPreprocessor`), HTML stripping, URL/Email/Currency/Phone semantic tokenization, stop-word filtering
        - **Data Visualization:** Plotly, Matplotlib, Seaborn
        - **Web Application Framework:** Streamlit
        - **Model Persistence:** Joblib & JSON Metadata
        - **Automated Testing:** Pytest

        ---

        ### 3. Machine Learning Algorithms Compared
        1. **Multinomial Naive Bayes (`MultinomialNB`):**
           - A probabilistic classifier based on Bayes' Theorem that models word/n-gram frequency distributions across classes.
           - Particularly fast and effective for high-dimensional text classification problems.
        2. **Logistic Regression (`LogisticRegression`):**
           - A discriminative linear classifier that learns optimal weights for each TF-IDF unigram and bigram using L2 regularization and balanced class weighting (`class_weight="balanced"`).
           - Calibrates class probabilities effectively and captures subtle interactions between promotional tokens.

        ---

        ### 4. Prevention of Data Leakage
        To guarantee scientifically honest evaluation:
        - The dataset is split into **80% training** and **20% testing** (`stratify=y`, `random_state=42`) **before** fitting the `TfidfVectorizer`.
        - Both preprocessing (`EmailTextPreprocessor`) and vectorization (`TfidfVectorizer`) are encapsulated inside a single Scikit-learn `Pipeline`, ensuring the vocabulary is learned exclusively from training messages.

        ---

        ### 5. Known Limitations & Future Enhancements
        - **Current Limitations:**
          - Analyzes email text content only; does not inspect SMTP headers (SPF, DKIM, DMARC) or file attachments.
          - Trained primarily on English-language messages; multilingual spam requires multilingual embeddings.
          - Adversarial obfuscation (e.g., image-only spam without body text) cannot be detected by text-only TF-IDF.
        - **Future Enhancements:**
          - Integration with transformer-based embeddings (DistilBERT / RoBERTa).
          - Live IMAP/Gmail API integration for automated inbox scanning.
          - URL domain reputation lookup and email header authentication checks.
        """
    )


# ==============================================================================
# MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main() -> None:
    st.set_page_config(
        page_title="MAILGUARD AI | Email Spam Detection",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🛡️ MAILGUARD AI")
        st.caption("Intelligent Email Security Suite")
        st.markdown("---")

        selected_page = st.radio(
            "Navigation Menu",
            options=[
                "🏠 Home Dashboard",
                "🔍 Email Spam Detection",
                "📂 Bulk Email Classification",
                "📊 Model Analytics",
                "🕒 Prediction History",
                "📘 About Project",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("**Quick System Status**")

    render_hero_header()

    # Safely load trained model and metadata
    try:
        pipeline, metadata = get_cached_model_and_metadata()
        with st.sidebar:
            st.success(f"Model Ready: {metadata.get('selected_model', 'Active')}")
            st.caption(f"Test F1: {metadata.get('best_model_metrics', {}).get('f1_score', 0) * 100:.2f}%")
    except ModelLoadError as exc:
        with st.sidebar:
            st.error("Model Not Loaded")
        st.error(f"⚠️ **Trained Model Unavailable:**\n\n{exc}")
        st.info(
            "**How to Train the Model:**\n"
            "Run the following command in your terminal from the project root directory:\n"
            "```bash\npython train.py\n```\n"
            "Or click the button below to run the training pipeline right now."
        )
        if st.button("⚙️ Train Model Now", type="primary"):
            with st.spinner("Training Multinomial Naive Bayes and Logistic Regression pipelines..."):
                try:
                    from src.training import train_and_save_models

                    train_and_save_models()
                    st.cache_resource.clear()
                    st.success("✅ Model trained and saved successfully! Reloading interface...")
                    st.rerun()
                except Exception as train_exc:
                    st.error(f"❌ Training failed: {train_exc}")
        return

    # Route to selected page
    if selected_page == "🏠 Home Dashboard":
        render_home_dashboard(metadata)
    elif selected_page == "🔍 Email Spam Detection":
        render_single_detection_page(pipeline, metadata)
    elif selected_page == "📂 Bulk Email Classification":
        render_bulk_classification_page(pipeline)
    elif selected_page == "📊 Model Analytics":
        render_model_analytics_page(metadata)
    elif selected_page == "🕒 Prediction History":
        render_prediction_history_page()
    elif selected_page == "📘 About Project":
        render_about_project_page(metadata)


if __name__ == "__main__":
    main()
