import os
import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# --- Classical model ---
from pmdarima import auto_arima

# --- Deep Learning ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

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
END_DATE   = "2025-07-31"  # last historical date available in the challenge
TICKER     = "TSLA"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Forecast horizons in trading days (~21 per month)
H6  = 6 * 21
H12 = 12 * 21
HORIZONS = {"6m": H6, "12m": H12}

WINDOW = 60        # LSTM lookback
N_SIM  = 200       # bootstrap paths for LSTM CIs
ALPHA  = 0.05      # 95% CI

# ----------------------------
# Utilities
# ----------------------------

def safe_price_column(df: pd.DataFrame) -> str:
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Adj_Close" in df.columns:
        return "Adj_Close"
    if "Close" in df.columns:
        return "Close"
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("adj close", "adj_close", "close"):
        if key in cols_lower:
            return cols_lower[key]
    raise KeyError("No suitable price column found. Expected one of: 'Adj Close' or 'Close'.")


def load_tsla_dataframe() -> pd.DataFrame:
    candidates = [
        "data_clean_TSLA.csv",
        os.path.join("data", "data_clean_TSLA.csv"),
        os.path.join("data", "TSLA.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
            df = df.sort_values("Date").reset_index(drop=True)
            return df
    if yf is None:
        raise FileNotFoundError("TSLA CSV not found and yfinance unavailable. Place data_clean_TSLA.csv in project root.")
    print("Downloading TSLA via yfinance (fallback)...")
    df = yf.download(TICKER, start=START_DATE, end=END_DATE)
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def business_days(start_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=n)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    eps  = 1e-9
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

# ----------------------------
# Load & prep data
# ----------------------------
all_df = load_tsla_dataframe()
all_df = all_df[(all_df["Date"] >= pd.to_datetime(START_DATE)) & (all_df["Date"] <= pd.to_datetime(END_DATE))]
all_df = all_df.sort_values("Date").reset_index(drop=True)

price_col = safe_price_column(all_df)
prices = all_df[["Date", price_col]].dropna().copy()
prices.rename(columns={price_col: "Price"}, inplace=True)
prices["LogPrice"] = np.log(prices["Price"])  # strictly positive

last_hist_date = prices["Date"].iloc[-1]

# ----------------------------
# Fit ARIMA on full historical log-prices
# ----------------------------
print("Fitting Auto-ARIMA on log-prices (full history up to END_DATE)...")
arima_model = auto_arima(
    y=prices["LogPrice"].values,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None, max_d=2,
    seasonal=False,
    stepwise=True,
    trace=False,
    error_action="ignore",
    suppress_warnings=True,
    with_intercept=True,
    information_criterion="aic",
)

# ----------------------------
# Fit LSTM on full historical log-prices
# ----------------------------
log_series = prices["LogPrice"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
log_scaled = scaler.fit_transform(log_series).reshape(-1)

X_train, y_train = [], []
for i in range(WINDOW, len(log_scaled)):
    X_train.append(log_scaled[i - WINDOW : i])
    y_train.append(log_scaled[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

lstm = keras.Sequential([
    layers.Input(shape=(WINDOW, 1)),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(32),
    layers.Dense(1)
])

lstm.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
_ = lstm.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, callbacks=[es], verbose=0)

# Build residuals for bootstrap (one-step)
# Predict the in-sample sequence targets to compute residuals in **log space** (after inverse-scaling)
train_preds_scaled = lstm.predict(X_train, verbose=0).reshape(-1)
train_preds_log = scaler.inverse_transform(train_preds_scaled.reshape(-1,1)).reshape(-1)
true_targets_log = scaler.inverse_transform(y_train.reshape(-1,1)).reshape(-1)
residuals_log = (true_targets_log - train_preds_log)

# ----------------------------
# Forecast functions
# ----------------------------

def arima_forecast_with_ci(h: int, alpha: float=ALPHA):
    fc_log, ci_log = arima_model.predict(n_periods=h, return_conf_int=True, alpha=alpha)
    # Move to price space (approx via exp transform)
    fc = np.exp(fc_log)
    lower = np.exp(ci_log[:, 0])
    upper = np.exp(ci_log[:, 1])
    return fc, lower, upper


def lstm_forecast_with_ci(h: int, n_sim: int=N_SIM, alpha: float=ALPHA):
    # Recursive point forecast in scaled space
    last_window = log_scaled[-WINDOW:].copy()
    preds_scaled = []
    for _ in range(h):
        x = last_window.reshape(1, WINDOW, 1)
        pred_scaled = lstm.predict(x, verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)
        last_window = np.append(last_window[1:], pred_scaled)
    point_log = scaler.inverse_transform(np.array(preds_scaled).reshape(-1,1)).reshape(-1)

    # Bootstrap simulations in log space by adding sampled residuals each step
    sims = np.zeros((n_sim, h))
    for s in range(n_sim):
        window = log_scaled[-WINDOW:].copy()
        sim_scaled = []
        for t in range(h):
            x = window.reshape(1, WINDOW, 1)
            pred_scaled = lstm.predict(x, verbose=0)[0, 0]
            # sample residual in log space
            eps = np.random.choice(residuals_log)
            # convert pred to log, add noise, then convert back to scaled by fitting scaler inverse -> forward
            pred_log = scaler.inverse_transform(np.array([[pred_scaled]])).reshape(-1)[0]
            noisy_log = pred_log + eps
            # back to scaled for recursion
            noisy_scaled = scaler.transform(np.array([[noisy_log]])).reshape(-1)[0]
            sim_scaled.append(noisy_scaled)
            window = np.append(window[1:], noisy_scaled)
        sims[s, :] = scaler.inverse_transform(np.array(sim_scaled).reshape(-1,1)).reshape(-1)

    # Convert log to price space
    point = np.exp(point_log)
    sims_price = np.exp(sims)
    lower = np.quantile(sims_price, q=alpha/2, axis=0)
    upper = np.quantile(sims_price, q=1-alpha/2, axis=0)
    return point, lower, upper

# ----------------------------
# Run forecasts and plot
# ----------------------------
for tag, H in HORIZONS.items():
    # timeline
    future_dates = business_days(prices["Date"].iloc[-1], H)

    # ARIMA
    ar_fc, ar_lo, ar_hi = arima_forecast_with_ci(H, alpha=ALPHA)

    # LSTM
    ls_fc, ls_lo, ls_hi = lstm_forecast_with_ci(H, n_sim=N_SIM, alpha=ALPHA)

    # --- Save summary CSVs (first/last values, CI width diagnostics) ---
    summary = pd.DataFrame({
        "Date": future_dates,
        "ARIMA_FC": ar_fc,
        "ARIMA_Lower": ar_lo,
        "ARIMA_Upper": ar_hi,
        "LSTM_FC": ls_fc,
        "LSTM_Lower": ls_lo,
        "LSTM_Upper": ls_hi,
    })
    # CI widths
    summary["ARIMA_CI_Width"] = summary["ARIMA_Upper"] - summary["ARIMA_Lower"]
    summary["LSTM_CI_Width"]  = summary["LSTM_Upper"] - summary["LSTM_Lower"]

    csv_path = os.path.join(OUTPUT_DIR, f"tsla_forecast_{tag}_summary.csv")
    summary.to_csv(csv_path, index=False)

    # --- Plot: history + forecasts with bands ---
    plt.figure(figsize=(13, 6))

    # Historical
    plt.plot(prices["Date"], prices["Price"], label="Historical")

    # ARIMA band
    plt.fill_between(future_dates, ar_lo, ar_hi, alpha=0.2, label="ARIMA 95% CI")
    plt.plot(future_dates, ar_fc, label="ARIMA Forecast")

    # LSTM band
    plt.fill_between(future_dates, ls_lo, ls_hi, alpha=0.2, label="LSTM 95% CI")
    plt.plot(future_dates, ls_fc, label="LSTM Forecast")

    plt.title(f"TSLA {tag.upper()} Forecast — ARIMA & LSTM with 95% CIs")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"tsla_forecast_{tag}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

print("\nDone. See 'outputs/tsla_forecast_6m.png' and 'outputs/tsla_forecast_12m.png' plus CSVs.")
