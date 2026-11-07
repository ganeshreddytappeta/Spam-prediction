"""
Visualization Module for MailGuard AI.

Provides Plotly and Matplotlib/Seaborn chart builders styled with a cohesive
cybersecurity dark-navy, slate, indigo/blue, emerald (Ham), and crimson (Spam)
visual palette.
"""

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Cohesive color palette
SPAM_COLOR = "#EF4444"       # Crimson Red for Spam alerts
HAM_COLOR = "#10B981"        # Emerald Green for Legitimate/Ham
ACCENT_BLUE = "#3B82F6"      # Royal Blue
ACCENT_PURPLE = "#6366F1"    # Indigo / Purple
NAVY_BG = "#0F172A"          # Deep Navy background
CARD_BG = "#1E293B"          # Slate Navy card background
TEXT_LIGHT = "#F8FAFC"       # Crisp White/Slate text


def create_class_distribution_chart(spam_count: int, ham_count: int) -> go.Figure:
    """
    Create an interactive Plotly Donut chart illustrating the Spam vs Ham
    distribution in the dataset.
    """
    labels = ["Legitimate (HAM)", "Spam (SPAM)"]
    values = [ham_count, spam_count]
    colors = [HAM_COLOR, SPAM_COLOR]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                marker=dict(colors=colors, line=dict(color=NAVY_BG, width=2)),
                textinfo="label+percent",
                textfont=dict(size=13, color=TEXT_LIGHT),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    total = spam_count + ham_count
    fig.update_layout(
        title=dict(
            text="Dataset Class Distribution (HAM vs SPAM)",
            font=dict(size=16, color=TEXT_LIGHT),
            x=0.02,
        ),
        annotations=[
            dict(
                text=f"<b>{total:,}</b><br>Total Emails",
                x=0.5,
                y=0.5,
                font=dict(size=15, color=TEXT_LIGHT),
                showarrow=False,
            )
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        margin=dict(t=50, b=20, l=20, r=20),
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color=TEXT_LIGHT),
        ),
    )
    return fig


def create_confusion_matrix_figure(cm: List[List[int]], model_name: str) -> plt.Figure:
    """
    Create a Matplotlib/Seaborn Confusion Matrix heatmap with both raw counts
    and row percentages, clearly annotating TN, FP, FN, and TP.
    """
    cm_array = np.array(cm, dtype=int)
    total = cm_array.sum() if cm_array.sum() > 0 else 1

    cell_titles = np.array(
        [
            ["True Negative (TN)\nCorrect Ham", "False Positive (FP)\nHam marked Spam"],
            ["False Negative (FN)\nMissed Spam", "True Positive (TP)\nCaught Spam"],
        ]
    )

    annot_labels = np.empty_like(cm_array, dtype=object)
    for i in range(2):
        for j in range(2):
            count = cm_array[i, j]
            pct = (count / total) * 100
            annot_labels[i, j] = f"{cell_titles[i, j]}\n\n{count:,}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=120)
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)

    sns.heatmap(
        cm_array,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted HAM (0)", "Predicted SPAM (1)"],
        yticklabels=["Actual HAM (0)", "Actual SPAM (1)"],
        linewidths=2,
        linecolor=NAVY_BG,
        annot_kws={"size": 10, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=12,
        fontweight="bold",
        color=TEXT_LIGHT,
        pad=14,
    )
    ax.set_xlabel("Model Prediction", fontsize=10, color=TEXT_LIGHT, labelpad=8)
    ax.set_ylabel("Ground Truth Label", fontsize=10, color=TEXT_LIGHT, labelpad=8)
    ax.tick_params(colors=TEXT_LIGHT, labelsize=9.5)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def create_model_comparison_chart(model_comparison: Dict[str, Dict[str, Any]]) -> go.Figure:
    """
    Create an interactive grouped bar chart comparing candidate Machine Learning
    models across Accuracy, Spam Precision, Spam Recall, and Spam F1-Score.
    """
    rows = []
    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Spam Precision",
        "recall": "Spam Recall",
        "f1_score": "Spam F1-Score",
    }

    for model_name, metrics in model_comparison.items():
        for key, display_name in metric_labels.items():
            rows.append(
                {
                    "Model": model_name,
                    "Metric": display_name,
                    "Score (%)": round(float(metrics.get(key, 0.0)) * 100, 2),
                }
            )

    df_plot = pd.DataFrame(rows)
    fig = px.bar(
        df_plot,
        x="Metric",
        y="Score (%)",
        color="Model",
        barmode="group",
        text="Score (%)",
        color_discrete_sequence=[ACCENT_BLUE, ACCENT_PURPLE],
    )
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(
        title=dict(
            text="Model Performance Comparison (Test Set)",
            font=dict(size=16, color=TEXT_LIGHT),
            x=0.02,
        ),
        yaxis=dict(
            range=[80, 104],
            title="Score (%)",
            gridcolor="rgba(255,255,255,0.08)",
            color=TEXT_LIGHT,
        ),
        xaxis=dict(title="", color=TEXT_LIGHT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title="",
            font=dict(color=TEXT_LIGHT),
        ),
        margin=dict(t=60, b=30, l=30, r=20),
        height=390,
    )
    return fig


def create_top_features_chart(
    features: List[Dict[str, Any]],
    title: str = "Top Spam-Indicating TF-IDF Terms",
    bar_color: str = SPAM_COLOR,
) -> go.Figure:
    """
    Create a horizontal bar chart showing the top TF-IDF terms/bigrams that drive
    Spam or Ham classification in the trained model.
    """
    if not features:
        fig = go.Figure()
        fig.update_layout(title="No feature weights available")
        return fig

    df_feat = pd.DataFrame(features[:15]).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=df_feat["weight"],
            y=df_feat["feature"],
            orientation="h",
            marker=dict(color=bar_color, line=dict(color=NAVY_BG, width=1)),
            text=[f"{w:.2f}" for w in df_feat["weight"]],
            textposition="auto",
            hovertemplate="Term: <b>%{y}</b><br>Model Weight: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT_LIGHT), x=0.02),
        xaxis=dict(
            title="Learned Model Weight / Log-Odds Impact",
            gridcolor="rgba(255,255,255,0.08)",
            color=TEXT_LIGHT,
        ),
        yaxis=dict(title="", color=TEXT_LIGHT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        margin=dict(t=45, b=35, l=110, r=20),
        height=410,
    )
    return fig


def create_probability_bar_chart(spam_prob: float, ham_prob: float) -> go.Figure:
    """
    Create a compact horizontal probability bar chart for single-email inference.
    """
    categories = ["Legitimate (HAM)", "Spam (SPAM)"]
    percentages = [round(ham_prob * 100, 2), round(spam_prob * 100, 2)]
    colors = [HAM_COLOR, SPAM_COLOR]

    fig = go.Figure(
        go.Bar(
            x=percentages,
            y=categories,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{p:.2f}%" for p in percentages],
            textposition="auto",
            hovertemplate="<b>%{y}</b>: %{x:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(
            text="Class Probability Distribution",
            font=dict(size=14, color=TEXT_LIGHT),
            x=0.0,
        ),
        xaxis=dict(
            range=[0, 100],
            title="Probability (%)",
            gridcolor="rgba(255,255,255,0.08)",
            color=TEXT_LIGHT,
        ),
        yaxis=dict(title="", color=TEXT_LIGHT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_LIGHT),
        margin=dict(t=35, b=25, l=120, r=20),
        height=200,
    )
    return fig

