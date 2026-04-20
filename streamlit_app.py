"""Streamlit dashboard for predictive maintenance model results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("Predictive Maintenance using Autoencoders")
st.write("App loaded successfully")


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_OPTIONS = ["LSTM", "GRU", "RNN"]


def get_model_paths(model_name: str) -> dict[str, Path]:
    model_key = model_name.lower()
    legacy_metrics = OUTPUT_DIR / "metrics.json"
    legacy_predictions = OUTPUT_DIR / "sequence_predictions.csv"
    legacy_history = OUTPUT_DIR / "history.csv"
    return {
        "metrics": OUTPUT_DIR / f"{model_key}_metrics.json",
        "train_metrics": OUTPUT_DIR / f"{model_key}_train_metrics.json",
        "predictions": OUTPUT_DIR / f"{model_key}_predictions.csv",
        "history": OUTPUT_DIR / f"{model_key}_history.csv",
        "legacy_metrics": legacy_metrics if model_name == "LSTM" else None,
        "legacy_predictions": legacy_predictions if model_name == "LSTM" else None,
        "legacy_history": legacy_history if model_name == "LSTM" else None,
        "training_loss_image": OUTPUT_DIR / f"{model_key}_training_loss.png",
        "legacy_training_loss_image": OUTPUT_DIR / "training_loss.png" if model_name == "LSTM" else None,
    }


def safe_load_json(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def safe_load_csv(path: Path) -> pd.DataFrame | None:
    try:
        if not path.exists():
            return None
        return pd.read_csv(path)
    except Exception:
        return None


def load_json_with_fallback(*paths: Path | None) -> dict | None:
    for path in paths:
        if path is None:
            continue
        loaded = safe_load_json(path)
        if loaded is not None:
            return loaded
    return None


def load_csv_with_fallback(*paths: Path | None) -> pd.DataFrame | None:
    for path in paths:
        if path is None:
            continue
        loaded = safe_load_csv(path)
        if loaded is not None:
            return loaded
    return None


def render_metric_cards(metrics: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.4f}")
    col2.metric("Precision", f"{float(metrics.get('precision', 0.0)):.4f}")
    col3.metric("Recall", f"{float(metrics.get('recall', 0.0)):.4f}")
    col4.metric("F1 Score", f"{float(metrics.get('f1_score', 0.0)):.4f}")


def render_training_metric_card(metrics: dict) -> None:
    st.metric("Accuracy", f"{float(metrics.get('accuracy', 0.0)):.4f}")
    st.caption(
        "Training data contains only normal samples, so precision, recall, and F1-score are not applicable."
    )


def create_train_eval_metrics_figure(train_metrics: dict | None, eval_metrics: dict | None, model_name: str) -> plt.Figure | None:
    if train_metrics is None or eval_metrics is None:
        return None

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1 Score"]
    train_values = [float(train_metrics.get(metric, 0.0)) for metric in metric_names]
    eval_values = [float(eval_metrics.get(metric, 0.0)) for metric in metric_names]
    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], train_values, width=width, label="Training")
    ax.bar([i + width / 2 for i in x], eval_values, width=width, label="Evaluation")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name} Training vs Evaluation Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def create_loss_figure(model_name: str) -> plt.Figure | None:
    model_paths = get_model_paths(model_name)
    history_df = load_csv_with_fallback(model_paths["history"], model_paths["legacy_history"])
    if history_df is None or history_df.empty:
        return None
    if "loss" not in history_df.columns or "val_loss" not in history_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history_df["loss"], label="Training Loss")
    ax.plot(history_df["val_loss"], label="Validation Loss")
    ax.set_title(f"{model_name} Loss Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def render_saved_loss_image(model_name: str) -> bool:
    model_paths = get_model_paths(model_name)
    for path in [model_paths["training_loss_image"], model_paths["legacy_training_loss_image"]]:
        if path is not None and path.exists():
            st.image(str(path), caption=f"{model_name} Training Loss")
            return True
    return False


def create_reconstruction_error_figure(model_name: str, predictions_df: pd.DataFrame, threshold: float) -> plt.Figure | None:
    if predictions_df.empty or "reconstruction_error" not in predictions_df.columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(predictions_df["reconstruction_error"], bins=30, alpha=0.7, color="steelblue")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6f}")
    ax.set_title(f"{model_name} Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def create_anomaly_figure(model_name: str, predictions_df: pd.DataFrame, threshold: float) -> plt.Figure | None:
    required_columns = {"sequence_index", "reconstruction_error", "predicted_label", "actual_label"}
    if predictions_df.empty or not required_columns.issubset(predictions_df.columns):
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        predictions_df["sequence_index"],
        predictions_df["reconstruction_error"],
        color="steelblue",
        label="Reconstruction Error",
    )
    ax.axhline(threshold, color="red", linestyle="--", linewidth=2, label="Threshold")

    predicted_anomalies = predictions_df[predictions_df["predicted_label"] == 1]
    actual_failures = predictions_df[predictions_df["actual_label"] == 1]

    if not actual_failures.empty:
        ax.scatter(
            actual_failures["sequence_index"],
            actual_failures["reconstruction_error"],
            color="orange",
            s=35,
            label="Actual Failures",
        )
    if not predicted_anomalies.empty:
        ax.scatter(
            predicted_anomalies["sequence_index"],
            predicted_anomalies["reconstruction_error"],
            color="crimson",
            marker="x",
            s=45,
            label="Detected Anomalies",
        )

    ax.set_title(f"{model_name} Anomaly Detection")
    ax.set_xlabel("Sequence Index")
    ax.set_ylabel("Reconstruction Error")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def create_comparison_figure(comparison_df: pd.DataFrame) -> plt.Figure | None:
    if comparison_df.empty:
        return None

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    missing = [metric for metric in metrics if metric not in comparison_df.columns]
    if missing:
        return None

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = comparison_df.set_index("Model")[metrics]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def create_training_time_figure(comparison_df: pd.DataFrame) -> plt.Figure | None:
    if comparison_df.empty or "Training Time (s)" not in comparison_df.columns:
        return None
    valid_df = comparison_df.dropna(subset=["Training Time (s)"]).copy()
    valid_df = valid_df[valid_df["Training Time (s)"] > 0]
    if valid_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(valid_df["Model"], valid_df["Training Time (s)"], color=["#4C78A8", "#F58518", "#54A24B"][: len(valid_df)])
    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.3)

    for bar, time_value in zip(bars, valid_df["Training Time (s)"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{time_value:.2f}s",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    return fig


def build_comparison_table() -> tuple[pd.DataFrame, dict | None]:
    comparison_rows = []
    comparison_json = safe_load_json(OUTPUT_DIR / "comparison_metrics.json")

    for model_name in MODEL_OPTIONS:
        model_paths = get_model_paths(model_name)
        metrics = load_json_with_fallback(model_paths["metrics"], model_paths["legacy_metrics"])
        if metrics is None:
            continue
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": float(metrics.get("accuracy", 0.0)),
                "Precision": float(metrics.get("precision", 0.0)),
                "Recall": float(metrics.get("recall", 0.0)),
                "F1 Score": float(metrics.get("f1_score", 0.0)),
                "Training Time (s)": metrics.get("training_time_seconds", metrics.get("training_time")),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df, comparison_json


selected_model = st.selectbox("Select Model", MODEL_OPTIONS)
model_paths = get_model_paths(selected_model)
metrics = load_json_with_fallback(model_paths["metrics"], model_paths["legacy_metrics"])
train_metrics = load_json_with_fallback(model_paths["train_metrics"])
predictions_df = load_csv_with_fallback(model_paths["predictions"], model_paths["legacy_predictions"])
metric_view = st.radio("Metric View", ["Evaluation Metrics", "Training Metrics"], horizontal=True)

st.header("Selected Model Results")

active_metrics = metrics if metric_view == "Evaluation Metrics" else train_metrics

if active_metrics is None:
    st.warning(f"Saved {metric_view.lower()} were not found for the selected model.")
else:
    if metric_view == "Training Metrics":
        render_training_metric_card(active_metrics)
    else:
        render_metric_cards(active_metrics)

    st.subheader("Confusion Matrix")
    confusion_matrix = active_metrics.get("confusion_matrix")
    if confusion_matrix is not None:
        if metric_view == "Training Metrics":
            st.caption("Training Data (Normal Only)")
        st.dataframe(pd.DataFrame(confusion_matrix), use_container_width=True)
    else:
        st.warning("Confusion matrix not found in saved metrics.")

if predictions_df is None:
    st.warning("Run training script first. Saved predictions were not found for the selected model.")
else:
    st.subheader("Prediction Samples")
    st.dataframe(predictions_df.head(20), use_container_width=True)

st.header("Graphs")

loss_fig = create_loss_figure(selected_model)
if loss_fig is not None:
    st.subheader("Training Loss Curve")
    st.pyplot(loss_fig)
    plt.close(loss_fig)
else:
    if render_saved_loss_image(selected_model):
        st.info("Showing saved training loss image because history CSV is not available.")
    else:
        st.warning("Training loss data not found. Save training history first.")
        st.warning("No saved training loss image was found either.")

threshold_value = float(metrics.get("threshold", 0.0)) if metrics is not None else 0.0
if predictions_df is not None and metrics is not None:
    error_fig = create_reconstruction_error_figure(selected_model, predictions_df, threshold_value)
    if error_fig is not None:
        st.subheader("Reconstruction Error")
        st.pyplot(error_fig)
        plt.close(error_fig)
    else:
        st.warning("Reconstruction error plot could not be created.")

    anomaly_fig = create_anomaly_figure(selected_model, predictions_df, threshold_value)
    if anomaly_fig is not None:
        st.subheader("Anomaly Detection")
        st.pyplot(anomaly_fig)
        plt.close(anomaly_fig)
    else:
        st.warning("Anomaly detection plot could not be created.")

metrics_compare_fig = create_train_eval_metrics_figure(train_metrics, metrics, selected_model)
if metrics_compare_fig is not None:
    st.subheader("Training vs Evaluation Metrics")
    st.pyplot(metrics_compare_fig)
    plt.close(metrics_compare_fig)

st.header("Comparison Dashboard")
comparison_df, comparison_json = build_comparison_table()

if comparison_df.empty:
    st.warning("Run training script first. No saved comparison data was found.")
else:
    st.subheader("Model Comparison Table")
    st.dataframe(comparison_df, use_container_width=True)

    if comparison_json is not None and "summary" in comparison_json:
        summary = comparison_json["summary"]
        st.write(
            f"Best model by F1-score: `{summary.get('better_model_by_f1', 'N/A')}` | "
            f"Best model by Recall: `{summary.get('better_model_by_recall', 'N/A')}` | "
            f"Fastest model: `{summary.get('faster_model_by_training_time', 'N/A')}`"
        )

    comparison_fig = create_comparison_figure(comparison_df)
    if comparison_fig is not None:
        st.subheader("Performance Comparison Chart")
        st.pyplot(comparison_fig)
        plt.close(comparison_fig)
    else:
        st.warning("Comparison chart could not be created.")

    training_time_fig = create_training_time_figure(comparison_df)
    if training_time_fig is not None:
        st.subheader("Training Time Comparison")
        st.pyplot(training_time_fig)
        plt.close(training_time_fig)
    else:
        st.info("Training time comparison is unavailable because saved metrics do not include training time yet.")

if comparison_df.empty and metrics is not None:
    st.subheader("Available Saved Results")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Model": selected_model,
                    "Accuracy": float(metrics.get("accuracy", 0.0)),
                    "Precision": float(metrics.get("precision", 0.0)),
                    "Recall": float(metrics.get("recall", 0.0)),
                    "F1 Score": float(metrics.get("f1_score", 0.0)),
                    "Training Time (s)": float(
                        metrics.get("training_time_seconds", metrics.get("training_time", 0.0))
                    ),
                }
            ]
        ),
        use_container_width=True,
    )
