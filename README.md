# 🛡️ MAILGUARD AI — Email Spam Detection System Using Machine Learning

> **"Intelligent Email Spam Detection Powered by Machine Learning"**  
> A complete, modular Supervised Machine Learning & Natural Language Processing (NLP) web application built with **Python**, **Scikit-learn**, and **Streamlit**.

---

## 📌 1. Overview

**MAILGUARD AI** is an end-to-end Email Spam Detection Web Application that analyzes email and message content to classify communications into:

- **SPAM (`1`)** — Unsolicited promotional messages, lottery/prize scams, phishing lures, and suspicious broadcasts.
- **HAM (`0` — NOT SPAM)** — Legitimate personal, academic, or professional communication.

Built with software engineering best practices, the project features automated dataset column discovery, a custom Scikit-learn NLP preprocessing transformer, leak-free TF-IDF feature extraction inside a unified `Pipeline`, comparative evaluation between **Multinomial Naive Bayes** and **Logistic Regression**, interactive model analytics, bulk CSV batch classification, and a full automated `pytest` test suite.

---

## 🎯 2. Problem Statement & Objectives

### Problem Statement
Email and messaging systems face a constant influx of spam, phishing links, and fraudulent financial lures. Static rule-based filters fail when spammers slightly modify wording or formatting. Conversely, overly aggressive filters risk flagging important legitimate messages as spam (**False Positives**).

### Primary Objectives
1. **Automated Data Validation & Cleaning:** Inspect raw CSV datasets automatically, detect text and label columns, normalize diverse label schemes (`spam`/`ham`, `1`/`0`, `junk`/`legitimate`), and remove duplicates and invalid rows.
2. **Domain-Aware NLP Preprocessing:** Clean HTML tags, normalize URLs (`urltoken`), email addresses (`emailtoken`), currency symbols (`currencytoken`), phone numbers (`phonetoken`), and excessive punctuation (`multiexclaimtoken`) so key spam indicators are preserved rather than blindly stripped.
3. **Leak-Free TF-IDF & Model Pipeline:** Encapsulate preprocessing, TF-IDF vectorization (`ngram_range=(1, 2)`), and classification inside a single `sklearn.pipeline.Pipeline` fit strictly on the 80% stratified training split.
4. **Multi-Metric Model Selection:** Compare **Multinomial Naive Bayes** and **Logistic Regression** using Spam F1-Score, Spam Precision, Spam Recall, Confusion Matrix, and ROC-AUC rather than accuracy alone.
5. **Interactive Cybersecurity Web Dashboard:** Provide single-email analysis, bulk CSV upload & download, model evaluation visual analytics, and session prediction history in Streamlit.

---

## ✨ 3. Key Features

- **🏠 Home Dashboard:** Real-time KPI cards, dataset preprocessing audit table, class distribution donut chart, and visual classification workflow.
- **🔍 Single Email Spam Detection:** One-click sample emails, input validation, SPAM/HAM warning/success banners, class probability breakdown bar chart, structural statistics, suspicious keyword badges, and downloadable JSON prediction reports.
- **📂 Bulk Email Classification:** Upload CSV batches up to 25 MB, select or auto-detect text columns, classify thousands of emails at once, inspect spam/ham counts, and download annotated CSV results.
- **📊 Model Analytics:** Compare Multinomial Naive Bayes vs. Logistic Regression, view side-by-side Confusion Matrix heatmaps, inspect top 15 Spam and Ham TF-IDF terms, and read clear explanations of Precision, Recall, False Positives, and False Negatives.
- **🕒 Prediction History:** Track all single-email checks during the current session and export the session log to CSV.
- **📘 About Project:** Architecture documentation, algorithm descriptions, limitations, and future roadmap.

---

## 🛠️ 4. Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Programming Language** | Python 3.10+ (tested on Python 3.12) |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (`Pipeline`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`) |
| **Natural Language Processing** | Regular Expressions (`re`), HTML Entity Unescaping, Custom `EmailTextPreprocessor`, Stop-word Filtering |
| **Data Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Application** | Streamlit |
| **Model Persistence** | Joblib (`best_model.joblib`), JSON (`model_metadata.json`) |
| **Unit Testing** | Pytest |

---

## 📊 5. Dataset Source & Preprocessing

- **Source:** [Kaggle SMS / Email Spam Collection Dataset](https://www.kaggle.com/datasets)
- **Location:** `data/emails.csv` (automatically copied from `spam.csv`)
- **Automated Preprocessing Pipeline (12 Steps):**
  1. Safely loads CSV across UTF-8, CP1252, and Latin-1 encodings.
  2. Records initial dimensions (**5,572 rows × 5 columns**).
  3. Automatically detects email content column (`v2`) and label column (`v1`), merging any text spilled across trailing unnamed columns.
  4. Identifies and removes missing values (**0 missing**).
  5. Removes empty or whitespace-only messages (**0 empty**).
  6. Detects and removes duplicate records (**403 duplicates removed**).
  7. Normalizes labels consistently to **`SPAM = 1`** and **`HAM = 0`**.
  8. Produces a clean dataset of **5,169 unique emails**:
     - **Legitimate (`HAM = 0`):** 4,516 emails (**87.37%**)
     - **Spam (`SPAM = 1`):** 653 emails (**12.63%**)

---

## 🏗️ 6. Project Architecture

```text
SL-Classification/
│
├── data/
│   └── emails.csv                 # Labeled email dataset (5,572 raw rows -> 5,169 cleaned)
│
├── models/
│   ├── best_model.joblib          # Complete saved Scikit-learn Pipeline (Preprocessor + TF-IDF + Classifier)
│   └── model_metadata.json        # Training stats, test metrics, confusion matrices & top TF-IDF terms
│
├── src/
│   ├── __init__.py                # Package initialization
│   ├── data_loader.py             # CSV encoding handling, column auto-detection & 12-step cleaning
│   ├── preprocessing.py           # NLP cleaning, spam-token preservation & EmailTextPreprocessor
│   ├── training.py                # Leak-free train/test split, Pipeline training & model persistence
│   ├── prediction.py              # Safe model loading, single-email & bulk CSV inference
│   ├── evaluation.py              # Accuracy, Precision, Recall, F1, ROC-AUC & Confusion Matrix
│   └── visualization.py           # Plotly & Seaborn/Matplotlib dark-themed visual charts
│
├── tests/
│   ├── test_preprocessing.py      # Unit tests for NLP cleaning, HTML/URL tokens & edge cases
│   ├── test_data_validation.py    # Unit tests for column detection, label mapping & duplicate removal
│   └── test_prediction.py         # Unit tests covering all 11 required prediction & error scenarios
│
├── app.py                         # Streamlit Web Application (MAILGUARD AI Dashboard)
├── train.py                       # CLI training & evaluation script
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules
```

---

## ⚙️ 7. Installation & Execution Instructions (Windows)

Run the following commands from the project root directory:

### 1. Create Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate Virtual Environment (Windows)
```powershell
venv\Scripts\activate
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Train and Evaluate Machine Learning Models
```powershell
python train.py
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```

### 6. Run Automated Unit Tests
```powershell
pytest -v
```

---

## 📈 8. Model Evaluation Methodology & Actual Results

### Prevention of Data Leakage
1. The cleaned dataset (**5,169 emails**) is split using an **80:20 stratified split** (`random_state=42`):
   - **Training Set (80%):** 4,135 emails (3,613 HAM, 522 SPAM)
   - **Testing Set (20%):** 1,034 emails (903 HAM, 131 SPAM)
2. `TfidfVectorizer(ngram_range=(1, 2), max_features=6000, min_df=2, sublinear_tf=True)` is fit **only** on the 4,135 training emails inside a Scikit-learn `Pipeline`.

### Actual Test Set Results (1,034 Unseen Emails)

| Model | Test Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Macro F1 | False Positives (FP) | False Negatives (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **⭐ Logistic Regression (Selected)** | **98.65%** | **95.35%** | **93.89%** | **94.62%** | **96.92%** | **6** | **8** |
| **Multinomial Naive Bayes** | 98.55% | 97.54% | 90.84% | 94.07% | 96.62% | 3 | 12 |

### Why Logistic Regression Was Selected
- Achieved the highest **Spam F1-Score (94.62% vs. 94.07% for Multinomial Naive Bayes)** and highest **Overall Accuracy (98.65%)**.
- Achieved **93.89% Spam Recall** and **95.35% Spam Precision**, producing only **6 False Positives** (out of 903 legitimate emails) and only **8 False Negatives** (out of 131 spam emails) on the held-out test set.
- **False Positives vs. False Negatives:**
  - **False Positive (FP):** A legitimate email is mistakenly classified as spam. Minimizing FP is crucial so users do not miss important work or personal emails.
  - **False Negative (FN):** A spam email is mistakenly classified as legitimate. High recall (93.89%) ensures nearly 94% of spam messages are caught automatically.

---

## 🖼️ 9. Application Screenshots (Placeholders)

> _Replace the placeholders below with screenshots captured while running `streamlit run app.py`:_

- **Home Dashboard:** `docs/screenshots/01_home_dashboard.png`
- **Single Email Spam Detection:** `docs/screenshots/02_spam_detection.png`
- **Bulk CSV Classification:** `docs/screenshots/03_bulk_classification.png`
- **Model Analytics & Confusion Matrix:** `docs/screenshots/04_model_analytics.png`

---

## ⚠️ 10. Known Limitations & Future Enhancements

### Known Limitations
- Evaluates message text only; does not inspect email transport headers (`SPF`, `DKIM`, `DMARC`) or binary attachments.
- Trained on English-language messages; non-English spam requires multilingual tokenization and training corpora.
- Image-only spam messages with minimal text cannot be detected without Optical Character Recognition (OCR).

### Future Enhancements
- Integrate Transformer-based contextual embeddings (**DistilBERT** / **RoBERTa**).
- Add live **IMAP / Gmail API** integration to scan an inbox automatically.
- Add real-time URL domain reputation checking and attachment sandboxing.
