import os
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Stats/ARIMA ---
from pmdarima import auto_arima

# --- Deep Learning ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Data ---
try:
    import yfinance as yf
except Exception:
    yf = None

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Configuration
# ----------------------------
START_DATE = "2015-07-01"
END_DATE   = "2025-07-31"
TRAIN_END  = "2023-12-31"  # inclusive
TEST_START = "2024-01-01"
TICKER     = "TSLA"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------

def safe_price_column(df: pd.DataFrame) -> str:
    """Return a robust price column name (Adj Close if present else Close)."""
    for c in df.columns:
        if isinstance(c, tuple):
            # Flatten MultiIndex if needed (yfinance sometimes returns this with group_by)
            pass
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Adj_Close" in df.columns:
        return "Adj_Close"
    if "Close" in df.columns:
        return "Close"
    # Last resort: case-insensitive search
    cols_lower = {col.lower(): col for col in df.columns}
    for key in ["adj close", "adj_close", "close"]:
        if key in cols_lower:
            return cols_lower[key]
    raise KeyError("No suitable price column found. Expected one of: 'Adj Close', 'Close'.")


def load_tsla_dataframe() -> pd.DataFrame:
    """Load TSLA OHLCV daily data. Prefer local clean CSV from Task 1; fallback to yfinance."""
    local_path_candidates = [
        "data_clean_TSLA.csv",
        os.path.join("data", "data_clean_TSLA.csv"),
        os.path.join("data", "TSLA.csv"),
    ]
    for path in local_path_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
            df = df.sort_values("Date").reset_index(drop=True)
            return df
    # Fallback: download (requires internet)
    if yf is None:
        raise FileNotFoundError(
            "No local TSLA CSV found and yfinance not available. Place data_clean_TSLA.csv in project root."
        )
    print("Downloading TSLA via yfinance (fallback)...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    df = df.reset_index()
    # Flatten potential MultiIndex columns
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def train_test_split_by_date(df: pd.DataFrame, date_col: str, train_end: str, test_start: str):
    train = df[df[date_col] <= pd.to_datetime(train_end)].copy()
    test  = df[df[date_col] >= pd.to_datetime(test_start)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    eps  = 1e-9
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def save_metrics_csv(rows: list, path: str):
    dfm = pd.DataFrame(rows)
    dfm.to_csv(path, index=False)
    return dfm

# ----------------------------
# 1) Load data
# ----------------------------
all_df = load_tsla_dataframe()

# Ensure within desired date range
all_df = all_df[(all_df["Date"] >= pd.to_datetime(START_DATE)) & (all_df["Date"] <= pd.to_datetime(END_DATE))]
all_df = all_df.sort_values("Date").reset_index(drop=True)

price_col = safe_price_column(all_df)

# Extract Date + price column, drop NA
prices = all_df[["Date", price_col]].dropna().copy()

# Rename and ensure numeric type
prices.rename(columns={price_col: "Price"}, inplace=True)
prices["Price"] = pd.to_numeric(prices["Price"], errors="coerce")

# Drop rows where price couldn't be converted to numeric
prices.dropna(subset=["Price"], inplace=True)

# Log transform (safe because all prices > 0)
prices["LogPrice"] = np.log(prices["Price"])
  # safe because Price > 0 for stocks

# Split
train_df, test_df = train_test_split_by_date(prices, "Date", TRAIN_END, TEST_START)

# Align arrays
y_train_log = train_df["LogPrice"].values
y_test_log  = test_df["LogPrice"].values

# ----------------------------
# Baseline: Naive (last observed value)
# ----------------------------
naive_forecasts_log = []
last_train_log = y_train_log[-1]
# Naive forecast repeats the last value for all test steps
naive_forecasts_log = np.array([last_train_log] * len(y_test_log))

# ----------------------------
# 2) ARIMA (auto_arima on LogPrice)
# ----------------------------
print("Fitting Auto-ARIMA on log prices (train)...")

arima_model = auto_arima(
    y=y_train_log,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None, max_d=2,         # let auto_arima decide differencing
    seasonal=False,
    stepwise=True,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    with_intercept=True,
    information_criterion="aic",
)

# Persist ARIMA summary
with open(os.path.join(OUTPUT_DIR, "arima_summary.txt"), "w") as f:
    f.write(str(arima_model.summary()))

# Forecast over test horizon (multi-step)
arima_forecasts_log = arima_model.predict(n_periods=len(y_test_log))

# ----------------------------
# 3) LSTM on LogPrice (scaled)
# ----------------------------
from sklearn.preprocessing import MinMaxScaler

WINDOW = 60  # lookback steps

# Scale using train only
scaler = MinMaxScaler(feature_range=(0, 1))
train_log_scaled = scaler.fit_transform(y_train_log.reshape(-1, 1)).reshape(-1)

# Create sequences for training
X_train, y_train = [], []
for i in range(WINDOW, len(train_log_scaled)):
    X_train.append(train_log_scaled[i - WINDOW : i])
    y_train.append(train_log_scaled[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape for LSTM [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build model
model = keras.Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dense(1)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es],
    verbose=1
)

# Recursive forecasting over test horizon
# We start from the last WINDOW observations of TRAIN (scaled), and predict forward one step at a time.
last_window = train_log_scaled[-WINDOW:].copy()
lstm_forecasts_scaled = []
for _ in range(len(y_test_log)):
    x = last_window.reshape(1, WINDOW, 1)
    pred_scaled = model.predict(x, verbose=0)[0, 0]
    lstm_forecasts_scaled.append(pred_scaled)
    # slide window: append pred, drop first
    last_window = np.append(last_window[1:], pred_scaled)

# Inverse scale back to log-space
lstm_forecasts_log = scaler.inverse_transform(np.array(lstm_forecasts_scaled).reshape(-1, 1)).reshape(-1)

# ----------------------------
# Convert log-forecasts back to price space
# ----------------------------
true_test_prices = np.exp(y_test_log)
naive_prices     = np.exp(naive_forecasts_log)
arima_prices     = np.exp(arima_forecasts_log)
lstm_prices      = np.exp(lstm_forecasts_log)

# ----------------------------
# Metrics
# ----------------------------
rows = []
for name, preds in [
    ("NAIVE", naive_prices),
    ("ARIMA", arima_prices),
    ("LSTM",  lstm_prices),
]:
    m = metrics(true_test_prices, preds)
    row = {"Model": name, **m}
    rows.append(row)

metrics_df = save_metrics_csv(rows, os.path.join(OUTPUT_DIR, "task2_metrics.csv"))
print("\nPerformance on TEST period (2024-01-01 to 2025-07-31):")
print(metrics_df)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(train_df["Date"], train_df["Price"], label="Train Actual")
plt.plot(test_df["Date"], test_df["Price"], label="Test Actual")
plt.plot(test_df["Date"], naive_prices, label="Naive Forecast")
plt.plot(test_df["Date"], arima_prices, label="ARIMA Forecast")
plt.plot(test_df["Date"], lstm_prices, label="LSTM Forecast")
plt.title(f"TSLA Forecasts — Train up to {TRAIN_END}, Test from {TEST_START} to {END_DATE}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task2_forecasts.png"), dpi=200)
plt.close()

# ----------------------------
# Save model artifacts
# ----------------------------
# LSTM model
model.save(os.path.join(OUTPUT_DIR, "lstm_model.keras"))

# Echo best ARIMA order
best_order = getattr(arima_model, "order", None)
best_seasonal_order = getattr(arima_model, "seasonal_order", None)
print(f"\nBest ARIMA order: {best_order}, seasonal_order: {best_seasonal_order}")

# ----------------------------
# Simple model comparison text
# ----------------------------
best = metrics_df.sort_values("RMSE").iloc[0]
print("\nBest model by RMSE:")
print(best)

with open(os.path.join(OUTPUT_DIR, "model_comparison.txt"), "w") as f:
    f.write("Performance on TEST period (2024-01-01 to 2025-07-31)\n")
    f.write(metrics_df.to_string(index=False))
    f.write("\n\nBest ARIMA order: " + str(best_order))
    f.write("\n")
    f.write("Best model by RMSE:\n" + best.to_string())

print("\nArtifacts saved to 'outputs/' directory.")