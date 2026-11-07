"""
WSGI Web Application Adapter for MAILGUARD AI.

Allows `app.py` to export a top-level PEP 3333 WSGI `app` and `application`
for cloud deployment platforms (such as Vercel, Render, Railway, or Gunicorn)
while sharing the exact same trained Scikit-learn model pipeline
(`models/best_model.joblib`) and inference logic (`src/prediction.py`) as the
Streamlit interface.
"""

import csv
import io
import json
from typing import Any, Callable, Dict, Iterable, List, Tuple

from src.prediction import (
    ModelLoadError,
    PredictionInputError,
    load_trained_model,
    predict_single_email,
)
from src.preprocessing import preprocess_text

TEXT_COLUMN_CANDIDATES = [
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

_CACHED_PIPELINE: Any = None
_CACHED_METADATA: Dict[str, Any] = {}


def _get_model_and_meta() -> Tuple[Any, Dict[str, Any]]:
    """Lazily load and cache the trained pipeline and metadata in memory."""
    global _CACHED_PIPELINE, _CACHED_METADATA
    if _CACHED_PIPELINE is None:
        _CACHED_PIPELINE, _CACHED_METADATA = load_trained_model()
    return _CACHED_PIPELINE, _CACHED_METADATA


def _json_response(
    start_response: Callable[[str, List[Tuple[str, str]]], Any],
    status: str,
    payload: Dict[str, Any],
) -> Iterable[bytes]:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("Cache-Control", "no-store"),
    ]
    start_response(status, headers)
    return [body]


def _read_request_body(environ: Dict[str, Any]) -> bytes:
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except (ValueError, TypeError):
        length = 0
    if length <= 0:
        return b""
    stream = environ.get("wsgi.input")
    if stream is None:
        return b""
    return stream.read(length)


def _render_dashboard_html(metadata: Dict[str, Any]) -> str:
    """Render the full 6-page MAILGUARD AI web dashboard backed by live Python API endpoints."""
    meta_json = json.dumps(metadata)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MAILGUARD AI | Email Spam Detection System</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body {{
      background-color: #0F172A;
      color: #F8FAFC;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .card {{
      background-color: #1E293B;
      border: 1px solid #334155;
      border-radius: 0.75rem;
    }}
    .hero-banner {{
      background: linear-gradient(135deg, #1E293B 0%, #0F172A 60%, #1E1B4B 100%);
      border: 1px solid #334155;
      border-left: 5px solid #6366F1;
      border-radius: 0.75rem;
    }}
  </style>
</head>
<body class="min-h-screen flex flex-col md:flex-row">
  <!-- Sidebar Navigation -->
  <aside class="w-full md:w-64 bg-[#1E293B] border-b md:border-b-0 md:border-r border-[#334155] p-5 shrink-0 flex flex-col justify-between">
    <div>
      <div class="flex items-center gap-2.5 mb-1">
        <span class="text-2xl">🛡️</span>
        <span class="text-lg font-extrabold tracking-wide text-white">MAILGUARD AI</span>
      </div>
      <p class="text-xs text-slate-400 mb-5">Intelligent Email Security Suite</p>
      <div class="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Navigation Menu</div>
      <nav class="space-y-1.5 text-sm">
        <button onclick="switchPage('home')" id="nav-home" class="w-full text-left px-3 py-2 rounded-lg font-semibold bg-indigo-600 text-white">🏠 Home Dashboard</button>
        <button onclick="switchPage('detect')" id="nav-detect" class="w-full text-left px-3 py-2 rounded-lg font-medium text-slate-300 hover:bg-slate-800">🔍 Email Spam Detection</button>
        <button onclick="switchPage('bulk')" id="nav-bulk" class="w-full text-left px-3 py-2 rounded-lg font-medium text-slate-300 hover:bg-slate-800">📂 Bulk Classification</button>
        <button onclick="switchPage('analytics')" id="nav-analytics" class="w-full text-left px-3 py-2 rounded-lg font-medium text-slate-300 hover:bg-slate-800">📊 Model Analytics</button>
        <button onclick="switchPage('history')" id="nav-history" class="w-full text-left px-3 py-2 rounded-lg font-medium text-slate-300 hover:bg-slate-800">🕒 Prediction History</button>
        <button onclick="switchPage('about')" id="nav-about" class="w-full text-left px-3 py-2 rounded-lg font-medium text-slate-300 hover:bg-slate-800">📘 About Project</button>
      </nav>
    </div>
    <div class="mt-6 pt-4 border-t border-[#334155] text-xs">
      <div class="px-3 py-2 rounded-lg bg-emerald-500/15 border border-emerald-500/40 text-emerald-300 font-semibold">
        ✅ Model Active: <span id="sidebar-model-name">Logistic Regression</span>
      </div>
      <div class="text-slate-400 mt-1.5 px-1" id="sidebar-f1">Test Spam F1: 94.62%</div>
    </div>
  </aside>

  <!-- Main Content Area -->
  <main class="flex-1 p-5 md:p-8 max-w-6xl mx-auto w-full space-y-6">
    <!-- Hero Header -->
    <div class="hero-banner p-5 shadow-lg">
      <h1 class="text-2xl font-extrabold text-white tracking-tight">🛡️ MAILGUARD AI</h1>
      <p class="text-sm text-slate-300 mt-1">Intelligent Email Spam Detection Powered by Machine Learning</p>
    </div>

    <!-- PAGE 1: HOME DASHBOARD -->
    <section id="page-home" class="space-y-6">
      <div>
        <h2 class="text-xl font-bold mb-1">🏠 Executive Security &amp; Model Overview</h2>
        <p class="text-sm text-slate-300">
          <strong>MAILGUARD AI</strong> analyzes email content using Natural Language Processing (NLP) and
          <strong>TF-IDF</strong> feature extraction inside a leak-free Scikit-learn Pipeline to classify messages as
          <strong>SPAM</strong> or <strong>HAM (Legitimate)</strong> in real time.
        </p>
      </div>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4" id="home-kpi-grid"></div>

      <div class="card p-5">
        <h3 class="text-sm font-bold uppercase tracking-wider text-slate-300 mb-3">⚙️ End-to-End Classification Workflow</h3>
        <div class="grid grid-cols-1 sm:grid-cols-5 gap-2.5 text-center text-xs">
          <div class="p-3 rounded-lg bg-slate-900/70 border border-indigo-500/50 font-semibold">1️⃣ Email Input<div class="text-slate-400 font-normal mt-1">Raw Text / CSV Batch</div></div>
          <div class="p-3 rounded-lg bg-slate-900/70 border border-indigo-500/50 font-semibold">2️⃣ Text Processing<div class="text-slate-400 font-normal mt-1">Clean, Normalize &amp; Tokenize</div></div>
          <div class="p-3 rounded-lg bg-slate-900/70 border border-indigo-500/50 font-semibold">3️⃣ TF-IDF Vectorizer<div class="text-slate-400 font-normal mt-1">Unigrams &amp; Bigrams (1,2)</div></div>
          <div class="p-3 rounded-lg bg-slate-900/70 border border-indigo-500/50 font-semibold">4️⃣ ML Classifier<div class="text-slate-400 font-normal mt-1" id="wf-model">Logistic Regression</div></div>
          <div class="p-3 rounded-lg bg-slate-900/70 border border-indigo-500/50 font-semibold">5️⃣ Prediction<div class="text-slate-400 font-normal mt-1">SPAM / HAM + Probability</div></div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div class="card p-5">
          <h3 class="text-base font-bold mb-3">📊 Dataset Class Distribution</h3>
          <div id="home-distribution-bars" class="space-y-4 pt-2"></div>
        </div>
        <div class="card p-5">
          <h3 class="text-base font-bold mb-3">🧹 Dataset Preprocessing Audit Log</h3>
          <div class="overflow-x-auto">
            <table class="w-full text-xs text-left">
              <tbody id="home-audit-table" class="divide-y divide-slate-700/60"></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- PAGE 2: EMAIL SPAM DETECTION -->
    <section id="page-detect" class="hidden space-y-5">
      <div class="card p-6 space-y-4">
        <div>
          <h2 class="text-xl font-bold">🔍 Real-Time Email Spam Detection</h2>
          <p class="text-sm text-slate-300 mt-1">
            Enter an email message below or click a sample button to classify it using the trained Scikit-learn pipeline.
          </p>
        </div>

        <div class="flex flex-wrap gap-2">
          <button onclick="loadSampleEmail(0)" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 border border-slate-600 transition">🎁 Promotional Prize Spam</button>
          <button onclick="loadSampleEmail(1)" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 border border-slate-600 transition">🏦 Phishing Account Alert</button>
          <button onclick="loadSampleEmail(2)" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 border border-slate-600 transition">💼 Legitimate Work Email (Ham)</button>
          <button onclick="loadSampleEmail(3)" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 border border-slate-600 transition">☕ Personal Lunch Message (Ham)</button>
          <button onclick="loadSampleEmail(4)" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 border border-slate-600 transition">🌐 HTML &amp; Link Marketing Spam</button>
        </div>

        <textarea id="email-input" rows="6" class="w-full rounded-xl bg-slate-900 border border-slate-700 p-3.5 text-sm text-white focus:outline-none focus:border-indigo-500" placeholder="Paste the email subject and body here..."></textarea>

        <div class="flex flex-wrap gap-3">
          <button onclick="submitSingleEmail()" id="predict-btn" class="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition">🛡️ Analyze &amp; Predict Email</button>
          <button onclick="clearSingleEmail()" class="px-4 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 text-sm transition">🗑️ Clear Input</button>
        </div>

        <div id="single-error-box" class="hidden rounded-xl border border-amber-500 bg-amber-500/15 p-4 text-sm text-amber-200"></div>

        <div id="single-result-container" class="hidden space-y-4 pt-2">
          <div id="single-banner" class="rounded-xl p-5 border-2"></div>

          <div class="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-700">
              <div class="text-xs text-slate-400 uppercase font-semibold">Predicted Class</div>
              <div id="res-label" class="text-xl font-bold mt-1"></div>
            </div>
            <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-700">
              <div class="text-xs text-slate-400 uppercase font-semibold">Spam Probability</div>
              <div id="res-spam-prob" class="text-xl font-bold mt-1"></div>
            </div>
            <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-700">
              <div class="text-xs text-slate-400 uppercase font-semibold">Model Confidence</div>
              <div id="res-confidence" class="text-xl font-bold mt-1"></div>
            </div>
          </div>

          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-700 space-y-2">
            <div class="text-xs font-semibold uppercase text-slate-400">Class Probability Breakdown</div>
            <div class="space-y-1.5 text-xs">
              <div class="flex justify-between"><span>Spam (SPAM = 1)</span><span id="bar-spam-label"></span></div>
              <div class="w-full h-3 bg-slate-800 rounded-full overflow-hidden"><div id="bar-spam" class="h-full bg-red-500"></div></div>
              <div class="flex justify-between pt-1"><span>Legitimate (HAM = 0)</span><span id="bar-ham-label"></span></div>
              <div class="w-full h-3 bg-slate-800 rounded-full overflow-hidden"><div id="bar-ham" class="h-full bg-emerald-500"></div></div>
            </div>
          </div>

          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-700 space-y-3">
            <div class="text-xs font-semibold uppercase text-slate-400">NLP Preprocessing &amp; Email Signals</div>
            <div class="grid grid-cols-2 sm:grid-cols-5 gap-2 text-xs" id="res-signals-grid"></div>
            <div>
              <div class="text-xs text-slate-400 mb-1">Suspicious / Promotional Keywords Present:</div>
              <div id="res-keywords" class="flex flex-wrap gap-1.5"></div>
            </div>
            <div>
              <div class="text-xs text-slate-400 mb-1">Normalized Text Passed to TF-IDF Vectorizer:</div>
              <pre id="res-cleaned-text" class="p-2.5 rounded bg-slate-950 text-xs text-indigo-300 overflow-x-auto"></pre>
            </div>
          </div>
        </div>
      </div>

      <div class="card p-4 border-l-4 border-l-blue-500 text-xs text-slate-300">
        ℹ️ <strong>Important Disclaimer:</strong> This application provides a Machine Learning-based classification and does not guarantee that an email is safe or malicious.
      </div>
    </section>

    <!-- PAGE 3: BULK EMAIL CLASSIFICATION -->
    <section id="page-bulk" class="hidden space-y-5">
      <div class="card p-6 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 class="text-xl font-bold">📂 Bulk Email Classification via CSV Upload</h2>
            <p class="text-sm text-slate-300 mt-1">Upload a CSV file containing multiple emails to classify the entire batch with the trained model.</p>
          </div>
          <button onclick="downloadSampleCSV()" class="text-xs px-3.5 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 font-semibold">📄 Download Sample CSV</button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
          <div class="md:col-span-2">
            <label class="block text-xs text-slate-400 mb-1">Select CSV File:</label>
            <input type="file" id="bulk-file-input" accept=".csv" class="w-full text-xs text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500" />
          </div>
          <div>
            <label class="block text-xs text-slate-400 mb-1">Text Column Name (optional auto-detect):</label>
            <input type="text" id="bulk-col-input" placeholder="Auto-detect (e.g., message, text, v2)" class="w-full rounded-lg bg-slate-900 border border-slate-700 p-2 text-xs text-white" />
          </div>
        </div>

        <button onclick="submitBulkCSV()" class="px-5 py-2.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm">🚀 Run Bulk Classification</button>

        <div id="bulk-error-box" class="hidden rounded-xl border border-red-500 bg-red-500/15 p-4 text-sm text-red-200"></div>

        <div id="bulk-result-container" class="hidden space-y-4 pt-3">
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3" id="bulk-summary-cards"></div>
          <div class="flex justify-between items-center">
            <h3 class="text-sm font-bold">Classified Dataset Preview</h3>
            <button onclick="downloadBulkResultCSV()" class="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold">📥 Download Classified CSV</button>
          </div>
          <div class="overflow-x-auto max-h-80 border border-slate-700 rounded-xl">
            <table class="w-full text-xs text-left" id="bulk-result-table"></table>
          </div>
        </div>
      </div>
    </section>

    <!-- PAGE 4: MODEL ANALYTICS -->
    <section id="page-analytics" class="hidden space-y-5">
      <div class="card p-6 space-y-4">
        <h2 class="text-xl font-bold">📊 Machine Learning Model Evaluation &amp; Analytics</h2>
        <div id="analytics-selection-reason" class="p-4 rounded-xl bg-indigo-500/15 border border-indigo-500/40 text-xs text-indigo-200"></div>
        <div class="overflow-x-auto">
          <table class="w-full text-xs text-left border-collapse" id="analytics-comp-table"></table>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-5" id="analytics-cm-grid"></div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div class="card p-5">
          <h3 class="text-sm font-bold mb-3">🔴 Top 12 Spam-Indicating TF-IDF Terms</h3>
          <div id="top-spam-list" class="space-y-2 text-xs"></div>
        </div>
        <div class="card p-5">
          <h3 class="text-sm font-bold mb-3">🟢 Top 12 Legitimate (Ham) TF-IDF Terms</h3>
          <div id="top-ham-list" class="space-y-2 text-xs"></div>
        </div>
      </div>
    </section>

    <!-- PAGE 5: PREDICTION HISTORY -->
    <section id="page-history" class="hidden space-y-5">
      <div class="card p-6 space-y-4">
        <div class="flex justify-between items-center">
          <div>
            <h2 class="text-xl font-bold">🕒 Session Prediction History</h2>
            <p class="text-xs text-slate-400 mt-0.5">Predictions made during this browser session (resets when session ends).</p>
          </div>
          <button onclick="predictionHistory=[]; renderHistory();" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-600 text-xs">🗑️ Clear History</button>
        </div>
        <div id="history-list" class="space-y-2 text-xs"></div>
      </div>
    </section>

    <!-- PAGE 6: ABOUT PROJECT -->
    <section id="page-about" class="hidden space-y-5">
      <div class="card p-6 space-y-4 text-sm text-slate-300 leading-relaxed">
        <h2 class="text-xl font-bold text-white">📘 About MAILGUARD AI</h2>
        <p><strong>Objective:</strong> Automatically detect unsolicited spam and phishing emails using Supervised Machine Learning and Natural Language Processing (NLP) without leaking test data into feature extraction.</p>
        <p><strong>Technology Stack:</strong> Python, Scikit-learn (<code>Pipeline</code>, <code>TfidfVectorizer</code>, <code>MultinomialNB</code>, <code>LogisticRegression</code>), Pandas, NumPy, Plotly, Matplotlib, Seaborn, Streamlit, Joblib, and Pytest.</p>
        <p><strong>Algorithms Compared:</strong> Multinomial Naive Bayes (probabilistic baseline) and Logistic Regression with balanced class weights and L2 regularization.</p>
      </div>
    </section>
  </main>

  <script>
    const METADATA = {meta_json};
    let predictionHistory = [];
    let lastBulkCSVOutput = "";

    const SAMPLE_EMAILS = [
      "URGENT! Congratulations, you have been selected to receive a £1000 CASH prize reward! To claim your guaranteed free bonus now, click here: http://www.prize-claim-win.com or call 09061701461 within 12 hours! Valid today only!!!",
      "FINAL NOTICE: Your online bank account has been temporarily suspended due to unusual activity. Verify your account password and credit card details immediately at http://secure-login-update.net/verify to avoid permanent closure. Free security update!",
      "Hi team, just a quick reminder that our sprint planning meeting is scheduled for tomorrow at 10:30 AM in Conference Room B. Please review the project backlog and let me know if we need to adjust any engineering tasks before the call. Thanks, Alex",
      "Hey! Are we still meeting for lunch around 1:00 PM after the lecture? I finished my database assignment early so I can bring the notes you asked for. See you at the campus cafe!",
      "<html><body><h2>FREE MEMBERSHIP OFFER!!!</h2><p>Congratulations! You won a 1 week FREE VIP voucher worth $500! <a href='https://free-voucher-deals.biz'>Click here now!</a> Reply CLAIM to 81010. T&amp;Cs apply.</p></body></html>"
    ];

    function switchPage(pageId) {{
      ['home', 'detect', 'bulk', 'analytics', 'history', 'about'].forEach(p => {{
        document.getElementById('page-' + p).classList.toggle('hidden', p !== pageId);
        document.getElementById('nav-' + p).className = p === pageId
          ? 'w-full text-left px-3 py-2 rounded-lg font-semibold bg-indigo-600 text-white'
          : 'w-full text-left px-3 py-2 rounded-lg font-medium text-slate-300 hover:bg-slate-800';
      }});
    }}

    function initDashboard() {{
      const ds = METADATA.dataset_statistics || {{}};
      const bm = METADATA.best_model_metrics || {{}};
      const selModel = METADATA.selected_model || "Logistic Regression";
      document.getElementById('sidebar-model-name').textContent = selModel;
      document.getElementById('wf-model').textContent = selModel;
      document.getElementById('sidebar-f1').textContent = 'Test Spam F1: ' + ((bm.f1_score || 0) * 100).toFixed(2) + '%';

      const kpis = [
        ["Total Cleaned Emails", (ds.final_rows || 0).toLocaleString(), `Raw: ${{(ds.initial_rows || 0).toLocaleString()}} (${{ds.duplicate_records_removed || 0}} duplicates removed)`],
        ["Total Spam Emails (1)", (ds.spam_count || 0).toLocaleString(), `${{ds.spam_percentage || 0}}% of dataset`],
        ["Total Legitimate Emails (0)", (ds.ham_count || 0).toLocaleString(), `${{ds.ham_percentage || 0}}% of dataset`],
        ["Selected ML Model", selModel, `TF-IDF Vocab: ${{(METADATA.vocabulary_size || 0).toLocaleString()}} terms`],
        ["Test Set Accuracy", ((bm.accuracy || 0) * 100).toFixed(2) + "%", `Spam Precision: ${{((bm.precision || 0) * 100).toFixed(2)}}%`],
        ["Spam Detection Recall", ((bm.recall || 0) * 100).toFixed(2) + "%", `Spam F1-Score: ${{((bm.f1_score || 0) * 100).toFixed(2)}}%`]
      ];
      document.getElementById('home-kpi-grid').innerHTML = kpis.map(([label, val, sub]) => `
        <div class="card p-4">
          <div class="text-xs uppercase tracking-wider text-slate-400 font-semibold">${{label}}</div>
          <div class="text-2xl font-bold mt-1 text-white">${{val}}</div>
          <div class="text-xs text-indigo-400 mt-1">${{sub}}</div>
        </div>
      `).join('');

      document.getElementById('home-distribution-bars').innerHTML = `
        <div>
          <div class="flex justify-between text-xs mb-1"><span>Legitimate (HAM = 0): ${{(ds.ham_count || 0).toLocaleString()}} emails</span><span class="font-bold text-emerald-400">${{ds.ham_percentage}}%</span></div>
          <div class="w-full h-4 bg-slate-800 rounded-full overflow-hidden"><div class="h-full bg-emerald-500" style="width:${{ds.ham_percentage}}%"></div></div>
        </div>
        <div>
          <div class="flex justify-between text-xs mb-1"><span>Spam (SPAM = 1): ${{(ds.spam_count || 0).toLocaleString()}} emails</span><span class="font-bold text-red-400">${{ds.spam_percentage}}%</span></div>
          <div class="w-full h-4 bg-slate-800 rounded-full overflow-hidden"><div class="h-full bg-red-500" style="width:${{ds.spam_percentage}}%"></div></div>
        </div>
      `;

      const auditRows = [
        ["Initial Dimensions", `${{(ds.initial_rows || 0).toLocaleString()}} rows × ${{ds.initial_columns || 0}} cols`],
        ["Detected Text / Label Columns", `'${{ds.detected_text_column}}' / '${{ds.detected_label_column}}'`],
        ["Missing / Empty Removed", `${{ds.total_missing_values || 0}} / ${{ds.empty_messages_removed || 0}}`],
        ["Duplicates Removed", (ds.duplicate_records_removed || 0).toLocaleString()],
        ["Final Clean Dataset", `${{(ds.final_rows || 0).toLocaleString()}} unique emails`]
      ];
      document.getElementById('home-audit-table').innerHTML = auditRows.map(([k, v]) => `
        <tr><td class="py-2 text-slate-400">${{k}}</td><td class="py-2 font-semibold text-right">${{v}}</td></tr>
      `).join('');

      // Analytics Page
      document.getElementById('analytics-selection-reason').textContent = METADATA.selection_reason || "";
      const comp = METADATA.model_comparison || {{}};
      document.getElementById('analytics-comp-table').innerHTML = `
        <thead>
          <tr class="border-b border-slate-700 text-slate-400">
            <th class="py-2">Model</th><th class="py-2">Accuracy</th><th class="py-2">Spam Precision</th><th class="py-2">Spam Recall</th><th class="py-2">Spam F1</th><th class="py-2">FP</th><th class="py-2">FN</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-slate-800">
          ${{Object.values(comp).map(m => `
            <tr class="${{m.model_name === selModel ? 'font-bold text-indigo-300' : ''}}">
              <td class="py-2.5">${{m.model_name === selModel ? '⭐ ' : ''}}${{m.model_name}}</td>
              <td>${{(m.accuracy * 100).toFixed(2)}}%</td>
              <td>${{(m.precision * 100).toFixed(2)}}%</td>
              <td>${{(m.recall * 100).toFixed(2)}}%</td>
              <td>${{(m.f1_score * 100).toFixed(2)}}%</td>
              <td>${{m.false_positives}}</td>
              <td>${{m.false_negatives}}</td>
            </tr>
          `).join('')}}
        </tbody>
      `;

      document.getElementById('analytics-cm-grid').innerHTML = Object.values(comp).map(m => `
        <div class="card p-5">
          <h4 class="text-sm font-bold mb-3">🧩 Confusion Matrix — ${{m.model_name}}</h4>
          <div class="grid grid-cols-2 gap-2.5 text-center text-xs">
            <div class="p-3 rounded-lg bg-emerald-500/15 border border-emerald-500/40">
              <div class="text-lg font-bold text-emerald-400">${{m.true_negatives}}</div>
              <div class="text-slate-300">True Negative (Correct Ham)</div>
            </div>
            <div class="p-3 rounded-lg bg-amber-500/15 border border-amber-500/40">
              <div class="text-lg font-bold text-amber-400">${{m.false_positives}}</div>
              <div class="text-slate-300">False Positive (Ham marked Spam)</div>
            </div>
            <div class="p-3 rounded-lg bg-amber-500/15 border border-amber-500/40">
              <div class="text-lg font-bold text-amber-400">${{m.false_negatives}}</div>
              <div class="text-slate-300">False Negative (Missed Spam)</div>
            </div>
            <div class="p-3 rounded-lg bg-indigo-500/15 border border-indigo-500/40">
              <div class="text-lg font-bold text-indigo-400">${{m.true_positives}}</div>
              <div class="text-slate-300">True Positive (Caught Spam)</div>
            </div>
          </div>
        </div>
      `).join('');

      const fi = METADATA.feature_insights || {{}};
      document.getElementById('top-spam-list').innerHTML = (fi.top_spam_features || []).slice(0, 12).map(f => `
        <div class="flex justify-between items-center border-b border-slate-800 pb-1">
          <span class="font-mono">${{f.feature}}</span>
          <span class="font-bold text-red-400">+${{f.weight.toFixed(2)}}</span>
        </div>
      `).join('');
      document.getElementById('top-ham-list').innerHTML = (fi.top_ham_features || []).slice(0, 12).map(f => `
        <div class="flex justify-between items-center border-b border-slate-800 pb-1">
          <span class="font-mono">${{f.feature}}</span>
          <span class="font-bold text-emerald-400">${{f.weight.toFixed(2)}}</span>
        </div>
      `).join('');

      renderHistory();
    }}

    function loadSampleEmail(idx) {{
      document.getElementById('email-input').value = SAMPLE_EMAILS[idx];
      submitSingleEmail();
    }}

    function clearSingleEmail() {{
      document.getElementById('email-input').value = '';
      document.getElementById('single-result-container').classList.add('hidden');
      document.getElementById('single-error-box').classList.add('hidden');
    }}

    async function submitSingleEmail() {{
      const text = document.getElementById('email-input').value;
      const errBox = document.getElementById('single-error-box');
      const resContainer = document.getElementById('single-result-container');
      errBox.classList.add('hidden');

      try {{
        const resp = await fetch('/api/predict', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ email_text: text }})
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          errBox.textContent = '⚠️ ' + (data.error || 'Validation failed');
          errBox.classList.remove('hidden');
          resContainer.classList.add('hidden');
          return;
        }}

        resContainer.classList.remove('hidden');
        const isSpam = data.predicted_class === 1;
        const banner = document.getElementById('single-banner');
        if (isSpam) {{
          banner.className = 'rounded-xl p-5 border-2 border-red-500 bg-red-500/15';
          banner.innerHTML = `<h3 class="text-lg font-bold text-red-300">🚨 CLASSIFICATION RESULT: SPAM DETECTED</h3>
            <p class="text-xs text-slate-200 mt-1">The Machine Learning model classified this email content as suspicious / unsolicited spam.</p>`;
        }} else {{
          banner.className = 'rounded-xl p-5 border-2 border-emerald-500 bg-emerald-500/15';
          banner.innerHTML = `<h3 class="text-lg font-bold text-emerald-300">✅ CLASSIFICATION RESULT: HAM (LEGITIMATE EMAIL)</h3>
            <p class="text-xs text-slate-200 mt-1">The Machine Learning model classified this email as normal legitimate communication.</p>`;
        }}

        const spamPct = ((data.spam_probability || 0) * 100).toFixed(2) + '%';
        const hamPct = ((data.ham_probability || 0) * 100).toFixed(2) + '%';
        document.getElementById('res-label').textContent = data.predicted_label;
        document.getElementById('res-spam-prob').textContent = spamPct;
        document.getElementById('res-confidence').textContent = (data.confidence_percent || 0).toFixed(2) + '%';

        document.getElementById('bar-spam-label').textContent = spamPct;
        document.getElementById('bar-spam').style.width = spamPct;
        document.getElementById('bar-ham-label').textContent = hamPct;
        document.getElementById('bar-ham').style.width = hamPct;

        const sig = data.signals || {{}};
        document.getElementById('res-signals-grid').innerHTML = [
          ['Words', sig.word_count || 0],
          ['Chars', sig.char_count || 0],
          ['URLs', sig.url_count || 0],
          ['Exclamations (!)', sig.exclamation_count || 0],
          ['Uppercase Ratio', ((sig.uppercase_ratio || 0) * 100).toFixed(1) + '%']
        ].map(([k, v]) => `<div class="p-2 rounded bg-slate-950 border border-slate-800"><div class="text-slate-400">${{k}}</div><div class="font-bold text-sm mt-0.5">${{v}}</div></div>`).join('');

        const kw = sig.matched_keywords || [];
        document.getElementById('res-keywords').innerHTML = kw.length
          ? kw.map(k => `<span class="px-2 py-0.5 rounded bg-red-500/20 border border-red-500 text-red-300 text-xs font-semibold">${{k}}</span>`).join('')
          : '<span class="text-xs text-slate-500">None detected</span>';
        document.getElementById('res-cleaned-text').textContent = data.cleaned_text || '';

        predictionHistory.unshift({{
          timestamp: new Date().toLocaleTimeString(),
          label: data.predicted_label,
          spam_prob: spamPct,
          preview: text.slice(0, 100) + (text.length > 100 ? '...' : '')
        }});
        renderHistory();
      }} catch (err) {{
        errBox.textContent = '❌ Request error: ' + err;
        errBox.classList.remove('hidden');
      }}
    }}

    function renderHistory() {{
      const el = document.getElementById('history-list');
      if (!predictionHistory.length) {{
        el.innerHTML = '<div class="text-slate-400">No emails analyzed in this session yet.</div>';
        return;
      }}
      el.innerHTML = predictionHistory.map(item => `
        <div class="p-3 rounded-lg bg-slate-900 border border-slate-700 flex justify-between items-center gap-3">
          <div><span class="font-bold ${{item.label === 'SPAM' ? 'text-red-400' : 'text-emerald-400'}}">[${{item.label}}]</span> ${{item.preview}}</div>
          <div class="text-slate-400 shrink-0">${{item.spam_prob}} • ${{item.timestamp}}</div>
        </div>
      `).join('');
    }}

    function downloadSampleCSV() {{
      const csv = "email_id,message\\n101,\\"URGENT! You have won a 1000 CASH prize! Call 09061701461\\"\\n102,\\"Hi team, reminder about our sprint planning meeting tomorrow at 10 AM.\\"\\n";
      const blob = new Blob([csv], {{ type: 'text/csv' }});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'sample_bulk_emails.csv';
      a.click();
    }}

    async function submitBulkCSV() {{
      const fileInput = document.getElementById('bulk-file-input');
      const colInput = document.getElementById('bulk-col-input').value.trim();
      const errBox = document.getElementById('bulk-error-box');
      const resBox = document.getElementById('bulk-result-container');
      errBox.classList.add('hidden');

      if (!fileInput.files || !fileInput.files[0]) {{
        errBox.textContent = '⚠️ Please choose a CSV file first.';
        errBox.classList.remove('hidden');
        return;
      }}
      const csvText = await fileInput.files[0].text();
      try {{
        const resp = await fetch('/api/bulk', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ csv_content: csvText, text_column: colInput || null }})
        }});
        const data = await resp.json();
        if (!resp.ok) {{
          errBox.textContent = '❌ ' + (data.error || 'Bulk classification error');
          errBox.classList.remove('hidden');
          resBox.classList.add('hidden');
          return;
        }}
        resBox.classList.remove('hidden');
        lastBulkCSVOutput = data.csv_output || "";
        const s = data.summary || {{}};
        document.getElementById('bulk-summary-cards').innerHTML = [
          ['Total Rows', s.total_rows || 0],
          ['Predicted SPAM', `${{s.spam_count || 0}} (${{s.spam_percentage || 0}}%)`],
          ['Predicted HAM', `${{s.ham_count || 0}} (${{s.ham_percentage || 0}}%)`],
          ['Empty / Skipped', s.empty_rows || 0]
        ].map(([k, v]) => `<div class="p-3.5 rounded-xl bg-slate-900 border border-slate-700"><div class="text-xs text-slate-400">${{k}}</div><div class="text-lg font-bold mt-0.5">${{v}}</div></div>`).join('');

        const rows = data.records || [];
        if (rows.length) {{
          const cols = Object.keys(rows[0]);
          document.getElementById('bulk-result-table').innerHTML = `
            <thead class="bg-slate-900 text-slate-400 border-b border-slate-700">
              <tr>${{cols.map(c => `<th class="p-2.5">${{c}}</th>`).join('')}}</tr>
            </thead>
            <tbody class="divide-y divide-slate-800">
              ${{rows.slice(0, 100).map(r => `<tr>${{cols.map(c => `<td class="p-2.5">${{r[c] ?? ''}}</td>`).join('')}}</tr>`).join('')}}
            </tbody>
          `;
        }}
      }} catch (e) {{
        errBox.textContent = '❌ Error: ' + e;
        errBox.classList.remove('hidden');
      }}
    }}

    function downloadBulkResultCSV() {{
      if (!lastBulkCSVOutput) return;
      const blob = new Blob([lastBulkCSVOutput], {{ type: 'text/csv' }});
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'mailguard_classified_results.csv';
      a.click();
    }}

    initDashboard();
  </script>
</body>
</html>"""


def wsgi_app(
    environ: Dict[str, Any],
    start_response: Callable[[str, List[Tuple[str, str]]], Any],
) -> Iterable[bytes]:
    """
    Standard PEP 3333 WSGI application callable exported as `app` and `application`
    in `app.py`.
    """
    path = environ.get("PATH_INFO", "/") or "/"
    method = (environ.get("REQUEST_METHOD", "GET") or "GET").upper()

    try:
        pipeline, metadata = _get_model_and_meta()
    except ModelLoadError as exc:
        return _json_response(
            start_response,
            "500 Internal Server Error",
            {"error": str(exc)},
        )

    # 1. Health check endpoint
    if path == "/api/health":
        return _json_response(
            start_response,
            "200 OK",
            {
                "status": "ok",
                "selected_model": metadata.get("selected_model"),
            },
        )

    # 2. Model metadata endpoint
    if path == "/api/metadata":
        return _json_response(start_response, "200 OK", metadata)

    # 3. Single email prediction endpoint
    if path == "/api/predict" and method == "POST":
        raw_body = _read_request_body(environ)
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except Exception:
            return _json_response(
                start_response,
                "400 Bad Request",
                {"error": "Invalid JSON request body."},
            )

        email_text = payload.get("email_text", "")
        try:
            result = predict_single_email(email_text, pipeline)
            return _json_response(start_response, "200 OK", result)
        except PredictionInputError as exc:
            return _json_response(
                start_response,
                "400 Bad Request",
                {"error": str(exc)},
            )
        except Exception as exc:
            return _json_response(
                start_response,
                "500 Internal Server Error",
                {"error": f"Prediction failed: {exc}"},
            )

    # 4. Bulk CSV prediction endpoint
    if path == "/api/bulk" and method == "POST":
        raw_body = _read_request_body(environ)
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            csv_content = (payload.get("csv_content") or "").strip()
            requested_col = payload.get("text_column")
            if not csv_content:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {"error": "Uploaded CSV file is empty."},
                )

            reader = csv.DictReader(io.StringIO(csv_content))
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {"error": "CSV file contains no valid header columns."},
                )

            rows = list(reader)
            if not rows:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {"error": "CSV file contains no data rows."},
                )

            target_col = requested_col
            if not target_col or target_col not in fieldnames:
                target_col = fieldnames[0]
                for col in fieldnames:
                    if str(col).strip().lower() in TEXT_COLUMN_CANDIDATES:
                        target_col = col
                        break

            total_rows = len(rows)
            valid_rows = 0
            empty_rows = 0
            spam_count = 0
            ham_count = 0
            annotated_rows: List[Dict[str, Any]] = []

            for r in rows:
                row_copy = dict(r)
                raw_val = (row_copy.get(target_col) or "").strip()
                if not raw_val or not preprocess_text(raw_val):
                    empty_rows += 1
                    row_copy["predicted_label"] = "EMPTY / INVALID"
                    row_copy["spam_probability"] = ""
                    row_copy["ham_probability"] = ""
                else:
                    valid_rows += 1
                    probs = pipeline.predict_proba([raw_val])[0]
                    ham_p = round(float(probs[0]), 4)
                    spam_p = round(float(probs[1]), 4)
                    label = "SPAM" if spam_p >= 0.5 else "HAM"
                    if label == "SPAM":
                        spam_count += 1
                    else:
                        ham_count += 1
                    row_copy["predicted_label"] = label
                    row_copy["spam_probability"] = spam_p
                    row_copy["ham_probability"] = ham_p
                annotated_rows.append(row_copy)

            if valid_rows == 0:
                return _json_response(
                    start_response,
                    "400 Bad Request",
                    {"error": f"All {total_rows} rows in column '{target_col}' are empty."},
                )

            out_fields = fieldnames + ["predicted_label", "spam_probability", "ham_probability"]
            out_buf = io.StringIO()
            writer = csv.DictWriter(out_buf, fieldnames=out_fields)
            writer.writeheader()
            writer.writerows(annotated_rows)

            summary = {
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "empty_rows": empty_rows,
                "spam_count": spam_count,
                "ham_count": ham_count,
                "spam_percentage": round((spam_count / valid_rows) * 100, 2),
                "ham_percentage": round((ham_count / valid_rows) * 100, 2),
            }

            return _json_response(
                start_response,
                "200 OK",
                {
                    "summary": summary,
                    "text_column": target_col,
                    "records": annotated_rows[:100],
                    "csv_output": out_buf.getvalue(),
                },
            )
        except Exception as exc:
            return _json_response(
                start_response,
                "400 Bad Request",
                {"error": f"Bulk classification failed: {exc}"},
            )

    # 5. Default GET / -> Serve full interactive MAILGUARD AI web dashboard
    html_bytes = _render_dashboard_html(metadata).encode("utf-8")
    headers = [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(html_bytes))),
    ]
    start_response("200 OK", headers)
    return [html_bytes]
