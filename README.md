# Predictive Maintenance Using Autoencoders

This project detects industrial machine anomalies using deep learning autoencoders on the **AI4I 2020 Predictive Maintenance Dataset**. It trains three sequence models and compares their anomaly detection performance:

- `LSTM Autoencoder`
- `GRU Autoencoder`
- `Simple RNN Autoencoder`

The project includes:

- data loading and preprocessing
- sliding-window sequence generation
- model training on normal data only
- anomaly detection using reconstruction error
- training and evaluation metrics
- a Streamlit dashboard for result visualization

## Project Structure

```text
miniproject/
├── ai4i2020.csv
├── predictive_maintenance_lstm_autoencoder.py
├── streamlit_app.py
├── requirements.txt
├── outputs/
└── README.md
```

## Dataset

Dataset used: **AI4I 2020 Predictive Maintenance Dataset**

Selected sensor features:

- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

Dropped columns:

- `UDI`
- `Product ID`
- `Type`

Target column:

- `Machine failure`

Important:

- training is done only on normal data where `Machine failure = 0`
- failure labels are used only for evaluation

## How It Works

Each model learns to reconstruct normal machine behavior from time-series sensor sequences.

Workflow:

1. Load and clean the dataset
2. Normalize selected features with `MinMaxScaler`
3. Split into normal training data and evaluation data
4. Create sliding-window sequences of length `30`
5. Train autoencoders on normal sequences only
6. Compute reconstruction error
7. Set anomaly threshold using:

```text
threshold = mean(training_error) + 3 * std(training_error)
```

8. Mark a sequence as anomaly if reconstruction error is greater than the threshold
9. Compare predictions with actual machine failure labels

## Models Implemented

### 1. LSTM Autoencoder

- LSTM encoder
- Dense bottleneck
- RepeatVector
- LSTM decoder
- TimeDistributed output

### 2. GRU Autoencoder

- GRU encoder
- Dense bottleneck
- RepeatVector
- GRU decoder
- TimeDistributed output

### 3. Simple RNN Autoencoder

- SimpleRNN encoder
- Dense bottleneck
- RepeatVector
- SimpleRNN decoder
- TimeDistributed output

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run Training

From the `miniproject` folder:

```bash
python predictive_maintenance_lstm_autoencoder.py --csv-path ai4i2020.csv
```

Or from the parent folder:

```bash
python miniproject/predictive_maintenance_lstm_autoencoder.py --csv-path miniproject/ai4i2020.csv
```

## Run Streamlit Dashboard

From the `miniproject` folder:

```bash
python -m streamlit run streamlit_app.py
```

Or from the parent folder:

```bash
python -m streamlit run miniproject/streamlit_app.py
```

## Saved Outputs

After training, the script automatically creates:

```text
outputs/
├── lstm_metrics.json
├── gru_metrics.json
├── rnn_metrics.json
├── lstm_train_metrics.json
├── gru_train_metrics.json
├── rnn_train_metrics.json
├── lstm_predictions.csv
├── gru_predictions.csv
├── rnn_predictions.csv
├── lstm_history.csv
├── gru_history.csv
├── rnn_history.csv
├── comparison_metrics.json
├── comparison_training_loss.png
├── comparison_validation_loss.png
├── comparison_error_distribution.png
├── comparison_metrics_bar.png
├── comparison_training_time.png
└── model-specific plots and `.keras` files
```

### Evaluation Metrics Files

Example:

- `outputs/lstm_metrics.json`
- `outputs/gru_metrics.json`
- `outputs/rnn_metrics.json`

These include:

- accuracy
- precision
- recall
- f1 score
- training time
- confusion matrix
- threshold

### Training Metrics Files

Example:

- `outputs/lstm_train_metrics.json`
- `outputs/gru_train_metrics.json`
- `outputs/rnn_train_metrics.json`

These are computed on training data using the same threshold.

Note:

- training data contains only normal samples
- therefore, in the dashboard, only **accuracy** is emphasized for training metrics
- precision, recall, and F1-score are not meaningful for normal-only training data

### Prediction Files

Example:

- `outputs/lstm_predictions.csv`
- `outputs/gru_predictions.csv`
- `outputs/rnn_predictions.csv`

Columns:

- `sequence_index`
- `actual_label`
- `predicted_label`
- `reconstruction_error`

### History Files

Example:

- `outputs/lstm_history.csv`
- `outputs/gru_history.csv`
- `outputs/rnn_history.csv`

Columns:

- `epoch`
- `loss`
- `val_loss`

## Streamlit Dashboard Features

The dashboard supports:

- model selection: `LSTM`, `GRU`, `RNN`
- evaluation metrics view
- training metrics view
- confusion matrix display
- training loss curve
- reconstruction error distribution
- anomaly detection plot
- model comparison table
- performance comparison chart
- training time comparison

If outputs are missing, the dashboard shows safe warnings instead of crashing.

## Notes

- The models are trained only on normal machine behavior.
- Thresholding is based on training reconstruction error.
- Training metrics use the same threshold as evaluation metrics.
- For training metrics, only accuracy is practically meaningful because the training set contains normal samples only.

## Example Use Case

This project is useful for:

- predictive maintenance demos
- anomaly detection coursework
- sequence autoencoder comparison studies
- industrial IoT fault detection experiments

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Future Improvements

- add model download and reuse inside Streamlit
- add ROC / PR analysis for evaluation data
- support configurable sequence length from the dashboard
- add upload-based dataset switching

## License

This project is for educational and research use.
