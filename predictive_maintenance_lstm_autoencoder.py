"""Predictive maintenance with LSTM, GRU, and RNN autoencoders on AI4I 2020 data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


DEFAULT_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
LABEL_COLUMN = "Machine failure"
DROP_COLUMNS = ["UDI", "Product ID", "Type"]
MODEL_NAMES = ["LSTM", "GRU", "RNN"]
BASE_DIR = Path(__file__).resolve().parent


def load_and_explore_data(csv_path: str | Path) -> pd.DataFrame:
    """Load the dataset and print a short exploratory summary."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=False)

    print("\n=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\n=== Basic Statistics ===")
    print(df.describe(include="all").transpose())

    print("\n=== Missing Values ===")
    print(df.isnull().sum())

    if df.isnull().sum().sum() > 0:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns

        if len(numeric_columns) > 0:
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        if len(categorical_columns) > 0:
            df[categorical_columns] = df[categorical_columns].fillna(
                df[categorical_columns].mode().iloc[0]
            )
        print("\nMissing values were found and imputed.")
    else:
        print("\nNo missing values found.")

    return df


def prepare_datasets(
    df: pd.DataFrame,
    features: list[str],
    label_column: str = LABEL_COLUMN,
    normal_eval_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Prepare train/evaluation splits and scale features."""
    cleaned_df = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()
    cleaned_df = cleaned_df[features + [label_column]].copy()

    normal_df = cleaned_df[cleaned_df[label_column] == 0].reset_index(drop=True)
    failure_df = cleaned_df[cleaned_df[label_column] == 1].reset_index(drop=True)

    split_index = int(len(normal_df) * (1 - normal_eval_ratio))
    split_index = max(split_index, 1)

    train_normal_df = normal_df.iloc[:split_index].copy()
    eval_normal_df = normal_df.iloc[split_index:].copy()
    evaluation_df = pd.concat([eval_normal_df, failure_df], axis=0, ignore_index=True)

    train_normal_df = ensure_float_features(train_normal_df, features, "train_normal_df")
    evaluation_df = ensure_float_features(evaluation_df, features, "evaluation_df")

    scaler = MinMaxScaler()
    train_normal_df[features] = scaler.fit_transform(train_normal_df[features])
    evaluation_df[features] = scaler.transform(evaluation_df[features])

    print("\n=== Post-Scaling Dtype Check ===")
    print("Training feature dtypes:")
    print(train_normal_df[features].dtypes)
    print("Evaluation feature dtypes:")
    print(evaluation_df[features].dtypes)

    print("\n=== Split Summary ===")
    print(f"Training normal rows: {len(train_normal_df)}")
    print(f"Evaluation normal rows: {len(eval_normal_df)}")
    print(f"Evaluation failure rows: {len(failure_df)}")
    print(f"Evaluation total rows: {len(evaluation_df)}")

    return train_normal_df, evaluation_df, scaler


def ensure_float_features(
    df: pd.DataFrame,
    features: list[str],
    dataset_name: str,
) -> pd.DataFrame:
    """Force feature columns to float before scaling to avoid dtype conflicts."""
    df = df.copy()

    print(f"\n=== Dtype Check: {dataset_name} before conversion ===")
    print(df[features].dtypes)

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    if df[features].isnull().sum().sum() > 0:
        print(f"Missing numeric values found in {dataset_name} after coercion. Filling with column medians.")
        df[features] = df[features].fillna(df[features].median())

    df[features] = df[features].astype(np.float32)

    print(f"\n=== Dtype Check: {dataset_name} after conversion ===")
    print(df[features].dtypes)

    return df


def create_sequences(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences and window-level labels."""
    values = feature_df.to_numpy(dtype=np.float32)
    label_values = labels.to_numpy(dtype=np.int32)

    sequences = []
    sequence_labels = []

    for start_idx in range(len(values) - sequence_length + 1):
        end_idx = start_idx + sequence_length
        window = values[start_idx:end_idx]
        window_label = int(label_values[start_idx:end_idx].max())
        sequences.append(window)
        sequence_labels.append(window_label)

    return np.array(sequences), np.array(sequence_labels)


def build_lstm_autoencoder(
    sequence_length: int,
    n_features: int,
    latent_dim: int = 16,
) -> Sequential:
    """Build and compile the LSTM autoencoder."""
    model = Sequential(
        [
            LSTM(64, activation="tanh", input_shape=(sequence_length, n_features), return_sequences=False),
            Dense(latent_dim, activation="relu"),
            RepeatVector(sequence_length),
            LSTM(64, activation="tanh", return_sequences=True),
            TimeDistributed(Dense(n_features)),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_gru_autoencoder(
    sequence_length: int,
    n_features: int,
    latent_dim: int = 16,
) -> Sequential:
    """Build and compile the GRU autoencoder."""
    model = Sequential(
        [
            GRU(64, activation="tanh", input_shape=(sequence_length, n_features), return_sequences=False),
            Dense(latent_dim, activation="relu"),
            RepeatVector(sequence_length),
            GRU(64, activation="tanh", return_sequences=True),
            TimeDistributed(Dense(n_features)),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_rnn_autoencoder(
    sequence_length: int,
    n_features: int,
    latent_dim: int = 16,
) -> Sequential:
    """Build and compile the Simple RNN autoencoder."""
    model = Sequential(
        [
            SimpleRNN(64, activation="tanh", input_shape=(sequence_length, n_features), return_sequences=False),
            Dense(latent_dim, activation="relu"),
            RepeatVector(sequence_length),
            SimpleRNN(64, activation="tanh", return_sequences=True),
            TimeDistributed(Dense(n_features)),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def train_model(
    model: Sequential,
    x_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float = 0.1,
):
    """Train the autoencoder on normal sequences only."""
    callbacks = [EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]

    history = model.fit(
        x_train,
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def compute_reconstruction_errors(model: Sequential, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct sequences and compute MSE per sequence."""
    reconstructed = model.predict(sequences, verbose=0)
    errors = np.mean(np.square(sequences - reconstructed), axis=(1, 2))
    return reconstructed, errors


def evaluate_predictions(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, zero_division=0),
        "recall": recall_score(true_labels, predicted_labels, zero_division=0),
        "f1_score": f1_score(true_labels, predicted_labels, zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels),
    }


def create_prediction_table(
    errors: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    prediction_table = pd.DataFrame(
        {
            "sequence_index": np.arange(len(errors)),
            "actual_label": true_labels,
            "reconstruction_error": errors,
            "threshold": threshold,
            "predicted_label": predicted_labels,
        }
    )
    prediction_table["prediction_name"] = np.where(
        prediction_table["predicted_label"] == 1, "Anomaly", "Normal"
    )
    return prediction_table


def plot_single_model_training_history(history, output_dir: Path, model_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_training_loss.png", dpi=150)
    plt.close()


def plot_single_model_error_distribution(
    train_errors: np.ndarray,
    eval_errors: np.ndarray,
    threshold: float,
    output_dir: Path,
    model_name: str,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(train_errors, bins=40, alpha=0.6, label="Training Errors")
    plt.hist(eval_errors, bins=40, alpha=0.6, label="Evaluation Errors")
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.6f}")
    plt.title(f"{model_name} Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_error_distribution.png", dpi=150)
    plt.close()


def plot_single_model_anomaly_predictions(
    errors: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    threshold: float,
    output_dir: Path,
    model_name: str,
) -> None:
    plt.figure(figsize=(12, 5))
    indices = np.arange(len(errors))

    plt.plot(indices, errors, label="Reconstruction Error", color="steelblue")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=2, label="Threshold")
    plt.scatter(indices[true_labels == 1], errors[true_labels == 1], color="orange", label="Actual Failures", s=35)
    plt.scatter(
        indices[predicted_labels == 1],
        errors[predicted_labels == 1],
        color="crimson",
        marker="x",
        label="Detected Anomalies",
        s=45,
    )

    plt.title(f"{model_name} Anomaly Detection on Evaluation Sequences")
    plt.xlabel("Sequence Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name.lower()}_anomaly_detection.png", dpi=150)
    plt.close()


def plot_comparison_training_histories(histories: dict, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model_name, history in histories.items():
        plt.plot(history.history["loss"], label=f"{model_name} Train")
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_training_loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    for model_name, history in histories.items():
        plt.plot(history.history["val_loss"], label=f"{model_name} Validation")
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_validation_loss.png", dpi=150)
    plt.close()


def plot_comparison_error_distributions(model_results: dict, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model_name, result in model_results.items():
        plt.hist(
            result["eval_errors"],
            bins=40,
            alpha=0.5,
            label=f"{model_name} Evaluation Errors",
        )
        plt.axvline(
            result["threshold"],
            linestyle="--",
            linewidth=2,
            label=f"{model_name} Threshold",
        )
    plt.title("Reconstruction Error Distribution Comparison")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_error_distribution.png", dpi=150)
    plt.close()


def plot_comparison_metrics(model_results: dict, output_dir: Path) -> None:
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score"]
    model_names = list(model_results.keys())
    x = np.arange(len(metrics_to_compare))
    width = 0.35

    plt.figure(figsize=(10, 5))
    for idx, model_name in enumerate(model_names):
        values = [model_results[model_name]["metrics"][metric] for metric in metrics_to_compare]
        offset = (idx - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model_name)

    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1 Score"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Metric Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_metrics_bar.png", dpi=150)
    plt.close()


def plot_training_time_comparison(model_results: dict, output_dir: Path) -> None:
    model_names = list(model_results.keys())
    training_times = [model_results[model_name]["training_time_seconds"] for model_name in model_names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, training_times, color=["#4C78A8", "#F58518", "#54A24B"][: len(model_names)])
    plt.ylabel("Training Time (seconds)")
    plt.title("Training Time Comparison")
    plt.grid(axis="y", alpha=0.3)

    for bar, time_value in zip(bars, training_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{time_value:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_training_time.png", dpi=150)
    plt.close()


def save_model_artifacts(
    output_path: Path,
    model_name: str,
    model: Sequential,
    history,
    train_errors: np.ndarray,
    eval_errors: np.ndarray,
    threshold: float,
    metrics: dict,
    train_metrics: dict,
    prediction_table: pd.DataFrame,
    training_time: float,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
) -> None:
    """Save model, metrics, predictions, and plots for one model."""
    model_key = model_name.lower()
    output_path.mkdir(parents=True, exist_ok=True)

    prediction_output = prediction_table[
        ["sequence_index", "actual_label", "predicted_label", "reconstruction_error"]
    ].copy()
    prediction_output.to_csv(output_path / f"{model_key}_predictions.csv", index=False)
    model.save(output_path / f"{model_key}_autoencoder.keras")

    metrics_to_save = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1_score": float(metrics["f1_score"]),
        "training_time": float(training_time),
        "training_time_seconds": float(training_time),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "threshold": float(threshold),
        "sequence_length": int(sequence_length),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "latent_dim": int(latent_dim),
    }
    with open(output_path / f"{model_key}_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics_to_save, metrics_file, indent=2)

    train_metrics_to_save = {
        "accuracy": float(train_metrics["accuracy"]),
        "precision": float(train_metrics["precision"]),
        "recall": float(train_metrics["recall"]),
        "f1_score": float(train_metrics["f1_score"]),
        "confusion_matrix": train_metrics["confusion_matrix"].tolist(),
        "threshold": float(threshold),
        "sequence_length": int(sequence_length),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "latent_dim": int(latent_dim),
    }
    with open(output_path / f"{model_key}_train_metrics.json", "w", encoding="utf-8") as train_metrics_file:
        json.dump(train_metrics_to_save, train_metrics_file, indent=2)

    history_df = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            "loss": history.history["loss"],
            "val_loss": history.history["val_loss"],
        }
    )
    history_df.to_csv(output_path / f"{model_key}_history.csv", index=False)

    plot_single_model_training_history(history, output_path, model_name)
    plot_single_model_error_distribution(
        train_errors,
        eval_errors,
        threshold,
        output_path,
        model_name,
    )
    plot_single_model_anomaly_predictions(
        eval_errors,
        prediction_table["actual_label"].to_numpy(),
        prediction_table["predicted_label"].to_numpy(),
        threshold,
        output_path,
        model_name,
    )


def train_and_evaluate_model(
    model_name: str,
    x_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    sequence_length: int,
    n_features: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
) -> Dict[str, object]:
    """Train one autoencoder model and evaluate its anomaly detection performance."""
    builders = {
        "LSTM": build_lstm_autoencoder,
        "GRU": build_gru_autoencoder,
        "RNN": build_rnn_autoencoder,
    }
    model = builders[model_name](
        sequence_length=sequence_length,
        n_features=n_features,
        latent_dim=latent_dim,
    )

    print(f"\n=== {model_name} Model Summary ===")
    model.summary()

    start_time = time.perf_counter()
    history = train_model(
        model=model,
        x_train=x_train,
        epochs=epochs,
        batch_size=batch_size,
    )
    training_time = time.perf_counter() - start_time

    _, train_errors = compute_reconstruction_errors(model, x_train)
    _, eval_errors = compute_reconstruction_errors(model, x_eval)

    threshold = train_errors.mean() + 3 * train_errors.std()
    y_train_expected = np.zeros(len(train_errors), dtype=int)
    predicted_train_labels = (train_errors > threshold).astype(int)
    train_metrics = evaluate_predictions(y_train_expected, predicted_train_labels)

    predicted_eval_labels = (eval_errors > threshold).astype(int)
    metrics = evaluate_predictions(y_eval, predicted_eval_labels)
    prediction_table = create_prediction_table(eval_errors, y_eval, predicted_eval_labels, threshold)

    print(f"\n=== {model_name} Threshold ===")
    print(f"Anomaly threshold: {threshold:.6f}")
    print(f"Training time: {training_time:.2f} seconds")

    print(f"\n=== {model_name} Evaluation Metrics ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    print(f"\n=== {model_name} Sample Predictions ===")
    print(prediction_table.head(10))

    detected_anomalies = prediction_table[prediction_table["predicted_label"] == 1]
    print(f"\n=== {model_name} Detected Anomalies (Top 10 by Error) ===")
    print(detected_anomalies.sort_values("reconstruction_error", ascending=False).head(10))

    print(f"\n=== {model_name} Training Metrics ===")
    print(f"Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"Precision: {train_metrics['precision']:.4f}")
    print(f"Recall   : {train_metrics['recall']:.4f}")
    print(f"F1-score : {train_metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(train_metrics["confusion_matrix"])

    return {
        "model": model,
        "history": history,
        "train_errors": train_errors,
        "eval_errors": eval_errors,
        "threshold": threshold,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "sample_predictions": prediction_table,
        "training_time_seconds": training_time,
    }


def print_model_comparison(model_results: dict) -> None:
    """Print a compact comparison summary for both models."""
    print("\n=== Model Comparison ===")
    for model_name, result in model_results.items():
        metrics = result["metrics"]
        print(
            f"{model_name} -> Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1_score']:.4f}, "
            f"Time: {result['training_time_seconds']:.2f}s"
        )

    better_model = max(model_results, key=lambda name: model_results[name]["metrics"]["f1_score"])
    best_recall_model = max(model_results, key=lambda name: model_results[name]["metrics"]["recall"])
    faster_model = min(model_results, key=lambda name: model_results[name]["training_time_seconds"])
    print(f"Better performing model (by F1-score): {better_model}")
    print(f"Best model by recall: {best_recall_model}")
    print(f"Faster model (by training time): {faster_model}")

    comparison_rows = []
    for model_name, result in model_results.items():
        metrics = result["metrics"]
        comparison_rows.append(
            {
                "Model": model_name,
                "Accuracy": round(float(metrics["accuracy"]), 4),
                "Precision": round(float(metrics["precision"]), 4),
                "Recall": round(float(metrics["recall"]), 4),
                "F1": round(float(metrics["f1_score"]), 4),
                "Training Time (s)": round(float(result["training_time_seconds"]), 2),
            }
        )
    print("\n=== Comparison Table ===")
    print(pd.DataFrame(comparison_rows).to_string(index=False))


def save_comparison_summary(
    output_path: Path,
    model_results: dict,
    sequence_length: int,
    epochs: int,
    batch_size: int,
    latent_dim: int,
) -> dict:
    """Save a top-level comparison summary JSON file."""
    comparison = {}
    for model_name, result in model_results.items():
        comparison[model_name] = {
            "accuracy": float(result["metrics"]["accuracy"]),
            "precision": float(result["metrics"]["precision"]),
            "recall": float(result["metrics"]["recall"]),
            "f1_score": float(result["metrics"]["f1_score"]),
            "confusion_matrix": result["metrics"]["confusion_matrix"].tolist(),
            "threshold": float(result["threshold"]),
            "training_time_seconds": float(result["training_time_seconds"]),
            "sequence_length": int(sequence_length),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "latent_dim": int(latent_dim),
        }

    better_model = max(comparison, key=lambda name: comparison[name]["f1_score"])
    best_recall_model = max(comparison, key=lambda name: comparison[name]["recall"])
    faster_model = min(comparison, key=lambda name: comparison[name]["training_time_seconds"])
    comparison["summary"] = {
        "better_model_by_f1": better_model,
        "better_model_by_recall": best_recall_model,
        "faster_model_by_training_time": faster_model,
    }

    with open(output_path / "comparison_metrics.json", "w", encoding="utf-8") as comparison_file:
        json.dump(comparison, comparison_file, indent=2)

    return comparison


def run_pipeline(
    csv_path: str | Path,
    sequence_length: int = 30,
    epochs: int = 25,
    batch_size: int = 32,
    normal_eval_ratio: float = 0.2,
    latent_dim: int = 16,
    output_dir: str | Path = BASE_DIR / "outputs",
    show_plots: bool = False,
) -> Dict[str, object]:
    """Execute the full predictive maintenance workflow for LSTM, GRU, and RNN models."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_and_explore_data(csv_path)
    train_df, eval_df, scaler = prepare_datasets(
        df=df,
        features=DEFAULT_FEATURES,
        label_column=LABEL_COLUMN,
        normal_eval_ratio=normal_eval_ratio,
    )

    x_train, y_train = create_sequences(
        train_df[DEFAULT_FEATURES],
        train_df[LABEL_COLUMN],
        sequence_length=sequence_length,
    )
    x_eval, y_eval = create_sequences(
        eval_df[DEFAULT_FEATURES],
        eval_df[LABEL_COLUMN],
        sequence_length=sequence_length,
    )

    if len(x_train) == 0 or len(x_eval) == 0:
        raise ValueError("Not enough rows to create sequences. Reduce sequence_length or provide more data.")

    print("\n=== Sequence Shapes ===")
    print(f"x_train shape: {x_train.shape}")
    print(f"x_eval shape: {x_eval.shape}")
    print(f"Training anomaly labels present: {np.unique(y_train)}")
    print(f"Evaluation label distribution: {dict(zip(*np.unique(y_eval, return_counts=True)))}")

    model_results: Dict[str, object] = {}
    for model_name in MODEL_NAMES:
        model_results[model_name] = train_and_evaluate_model(
            model_name=model_name,
            x_train=x_train,
            x_eval=x_eval,
            y_eval=y_eval,
            sequence_length=sequence_length,
            n_features=len(DEFAULT_FEATURES),
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
        )
        save_model_artifacts(
            output_path=output_path,
            model_name=model_name,
            model=model_results[model_name]["model"],
            history=model_results[model_name]["history"],
            train_errors=model_results[model_name]["train_errors"],
            eval_errors=model_results[model_name]["eval_errors"],
            threshold=model_results[model_name]["threshold"],
            metrics=model_results[model_name]["metrics"],
            train_metrics=model_results[model_name]["train_metrics"],
            prediction_table=model_results[model_name]["sample_predictions"],
            training_time=model_results[model_name]["training_time_seconds"],
            sequence_length=sequence_length,
            epochs=epochs,
            batch_size=batch_size,
            latent_dim=latent_dim,
        )

    plot_comparison_training_histories(
        {name: result["history"] for name, result in model_results.items()},
        output_path,
    )
    plot_comparison_error_distributions(model_results, output_path)
    plot_comparison_metrics(model_results, output_path)
    plot_training_time_comparison(model_results, output_path)

    print_model_comparison(model_results)
    comparison = save_comparison_summary(
        output_path=output_path,
        model_results=model_results,
        sequence_length=sequence_length,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
    )

    return {
        "model_results": model_results,
        "comparison": comparison,
        "scaler": scaler,
        "metrics": model_results["LSTM"]["metrics"],
        "sample_predictions": model_results["LSTM"]["sample_predictions"],
        "model": model_results["LSTM"]["model"],
        "history": model_results["LSTM"]["history"],
        "train_errors": model_results["LSTM"]["train_errors"],
        "eval_errors": model_results["LSTM"]["eval_errors"],
        "threshold": model_results["LSTM"]["threshold"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predictive maintenance with LSTM, GRU, and RNN autoencoders.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="miniproject/ai4i2020.csv",
        help="Path to the AI4I 2020 CSV dataset.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Sliding window sequence length.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--normal-eval-ratio",
        type=float,
        default=0.2,
        help="Fraction of normal rows reserved for evaluation.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Size of the dense bottleneck representation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "outputs"),
        help="Directory for saved models, plots, histories, metrics, and predictions.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_pipeline(
        csv_path=args.csv_path,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        normal_eval_ratio=args.normal_eval_ratio,
        latent_dim=args.latent_dim,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
