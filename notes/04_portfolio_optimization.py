import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyPortfolioOpt
from pypfopt import EfficientFrontier, risk_models

try:
    import yfinance as yf
except Exception:
    yf = None

START_DATE = "2015-07-01"
END_DATE   = "2025-07-31"
TICKERS    = ["TSLA", "BND", "SPY"]
TRADING_DAYS = 252

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Helpers
# ----------------------------

def safe_price_column(df: pd.DataFrame) -> str:
    for col in ("Adj Close", "Adj_Close", "Close"):
        if col in df.columns:
            return col
    cols_lower = {c.lower(): c for c in df.columns}
    for key in ("adj close", "adj_close", "close"):
        if key in cols_lower:
            return cols_lower[key]
    raise KeyError("No suitable price column found.")


def load_asset(ticker: str) -> pd.DataFrame:
    # Prefer Task 1 cleaned CSVs
    candidates = [
        f"data_clean_{ticker}.csv",
        os.path.join("data", f"data_clean_{ticker}.csv"),
        os.path.join("data", f"{ticker}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
            df = df.sort_values("Date").reset_index(drop=True)
            return df
    if yf is None:
        raise FileNotFoundError(f"Missing {ticker} CSV and yfinance unavailable. Place data_clean_{ticker}.csv.")
    print(f"Downloading {ticker} via yfinance (fallback)...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE).reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    price_col = safe_price_column(df)
    out = df[["Date", price_col]].dropna().copy()
    out.rename(columns={price_col: "Price"}, inplace=True)
    out.sort_values("Date", inplace=True)
    out["Return"] = out["Price"].pct_change()
    return out.dropna().reset_index(drop=True)


def annualize_daily_mean(mean_daily: float) -> float:
    # Approx geometric: (1+mu)^252 - 1 is better, but we will use geometric below when path-based
    return (1 + mean_daily) ** TRADING_DAYS - 1


def tsla_expected_from_forecast(horizon: str) -> float:
    """Compute TSLA expected annual return using Task 3 forecast CSV.
    We derive an average daily rate across the horizon and annualize geometrically.
    Fallback: historical mean annualized if CSV not found.
    """
    csv_path = os.path.join(OUTPUT_DIR, f"tsla_forecast_{horizon}_summary.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] {csv_path} not found. Falling back to historical mean for TSLA.")
        tsla_df = load_asset("TSLA")
        r = daily_returns(tsla_df)
        mu_daily = r["Return"].mean()
        return (1 + mu_daily) ** TRADING_DAYS - 1

    df = pd.read_csv(csv_path)
    # Prefer the model that won Task 2 if available; else use ARIMA forecast
    metrics_path = os.path.join(OUTPUT_DIR, "task2_metrics.csv")
    model_col = "ARIMA_FC"  # default
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path)
        best = m.sort_values("RMSE").iloc[0]["Model"].upper()
        if best == "LSTM":
            model_col = "LSTM_FC"
        else:
            model_col = "ARIMA_FC"

    # Derive per-step daily returns from forecast path (price_t / price_{t-1} - 1)
    prices = df[model_col].astype(float).values
    # simple guard
    prices = prices[~np.isnan(prices)]
    if len(prices) < 2:
        print("[WARN] Not enough forecast points; falling back to historical.")
        tsla_df = load_asset("TSLA")
        r = daily_returns(tsla_df)
        mu_daily = r["Return"].mean()
        return (1 + mu_daily) ** TRADING_DAYS - 1

    daily_rets = prices[1:] / prices[:-1] - 1
    mu_daily = np.mean(daily_rets)
    mu_annual = (1 + mu_daily) ** TRADING_DAYS - 1
    return float(mu_annual)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", choices=["6m", "12m"], default="12m", help="Use Task 3 6m or 12m forecast for TSLA expected return")
    parser.add_argument("--risk_free", type=float, default=0.02, help="Annual risk-free rate (e.g., 0.02 = 2%)")
    args = parser.parse_args()

    # Load assets
    frames = {}
    for t in TICKERS:
        frames[t] = load_asset(t)

    # Build returns table
    returns = {}
    for t in TICKERS:
        r = daily_returns(frames[t])
        returns[t] = r.set_index("Date")["Return"]
    returns_df = pd.DataFrame(returns).dropna(how="any")

    # Expected returns (annual)
    # TSLA from forecast; BND & SPY from historical mean (annualized)
    mu = {}
    mu["TSLA"] = tsla_expected_from_forecast(args.horizon)
    mu["BND"]  = annualize_daily_mean(returns_df["BND"].mean())
    mu["SPY"]  = annualize_daily_mean(returns_df["SPY"].mean())

    mu_vec = pd.Series(mu)[TICKERS]
    mu_vec.to_csv(os.path.join(OUTPUT_DIR, "expected_returns_vector.csv"), header=["ExpectedAnnualReturn"])

    # Covariance (daily) → annualize by *252
    S_daily = returns_df.cov()
    S = S_daily * TRADING_DAYS

    # Efficient Frontier
    ef = EfficientFrontier(mu_vec.values, S, weight_bounds=(0, 1))

    # Maximum Sharpe
    w_msr = ef.max_sharpe(risk_free_rate=args.risk_free)
    perf_msr = ef.portfolio_performance(verbose=False, risk_free_rate=args.risk_free)
    ef.clean_weights()
    weights_msr = pd.Series(w_msr, index=TICKERS)

    # Minimum Volatility (re-initialize EF to reset state)
    ef2 = EfficientFrontier(mu_vec.values, S, weight_bounds=(0, 1))
    w_minv = ef2.min_volatility()
    perf_minv = ef2.portfolio_performance(verbose=False, risk_free_rate=args.risk_free)
    ef2.clean_weights()
    weights_minv = pd.Series(w_minv, index=TICKERS)

    # Construct frontier by scanning target returns
    # We'll sample a grid of target returns between min and max of mu
    t_min, t_max = float(mu_vec.min()), float(mu_vec.max())
    grid = np.linspace(t_min, t_max, 50)
    frontier_pts = []
    for target in grid:
        ef_tmp = EfficientFrontier(mu_vec.values, S, weight_bounds=(0, 1))
        try:
            ef_tmp.efficient_return(target)
            ret, vol, sharpe = ef_tmp.portfolio_performance(risk_free_rate=args.risk_free, verbose=False)
            frontier_pts.append((vol, ret))
        except Exception:
            continue

    # Plot
    plt.figure(figsize=(10, 6))
    if frontier_pts:
        vols, rets = zip(*frontier_pts)
        plt.plot(vols, rets, label="Efficient Frontier")

    # Mark MSR and MinVol
    vol_msr = perf_msr[1]
    ret_msr = perf_msr[0]
    vol_minv = perf_minv[1]
    ret_minv = perf_minv[0]

    plt.scatter([vol_msr], [ret_msr], marker="*", s=200, label="Max Sharpe")
    plt.scatter([vol_minv], [ret_minv], marker="o", s=120, label="Min Volatility")

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Expected Annual Return")
    plt.title(f"Efficient Frontier (TSLA from {args.horizon} forecast)\nRisk-free={args.risk_free:.2%}")
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "efficient_frontier.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Save metrics & weights
    rows = [
        {
            "Portfolio": "Max Sharpe",
            "TSLA": weights_msr["TSLA"],
            "BND":  weights_msr["BND"],
            "SPY":  weights_msr["SPY"],
            "Expected Return": ret_msr,
            "Volatility": vol_msr,
            "Sharpe": perf_msr[2],
        },
        {
            "Portfolio": "Min Volatility",
            "TSLA": weights_minv["TSLA"],
            "BND":  weights_minv["BND"],
            "SPY":  weights_minv["SPY"],
            "Expected Return": ret_minv,
            "Volatility": vol_minv,
            "Sharpe": perf_minv[2],
        },
    ]
    out_df = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "optimal_portfolios.csv")
    out_df.to_csv(out_path, index=False)

    # Print concise recommendation (choose Max Sharpe by default)
    choice = rows[0]
    print("\nRecommended Portfolio: Max Sharpe (Tangency)")
    print(pd.Series(choice))
    print(f"\nArtifacts saved: {fig_path}, {out_path}")

if __name__ == "__main__":
    main()