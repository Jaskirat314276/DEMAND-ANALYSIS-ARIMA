"""End-to-end ARIMA / SARIMA workflow on the Perrin Frères champagne dataset.

Mirrors the steps in Untitled.ipynb but uses the current statsmodels / pandas APIs
and saves diagnostic plots + a 24-month forecast to ./results/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.plotting import autocorrelation_plot
from pandas.tseries.offsets import DateOffset
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

CSV = Path(__file__).parent / "perrin-freres-monthly-champagne-.csv"
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)


def adfuller_test(series, label):
    result = adfuller(series)
    names = ["ADF Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"]
    print(f"\n--- ADF Test on {label} ---")
    for value, name in zip(result, names):
        print(f"{name} : {value}")
    if result[1] <= 0.05:
        print("=> Stationary (reject H0)")
    else:
        print("=> Non-stationary (fail to reject H0)")
    return result[1]


def main():
    print("Step 1: Load + clean data")
    df = pd.read_csv(CSV)
    df.columns = ["Month", "demand"]
    # Drop trailing footer rows (NaN row + source-attribution row)
    df = df.dropna().reset_index(drop=True)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")
    df.index.freq = "MS"
    print(df.describe())

    # Step 2: visualise raw series
    ax = df["demand"].plot(figsize=(12, 5), title="Monthly Champagne Demand (1964-1972)")
    ax.set_ylabel("demand")
    plt.tight_layout()
    plt.savefig(RESULTS / "01_raw_series.png", dpi=120)
    plt.close()

    # Step 3: stationarity tests
    adfuller_test(df["demand"], "raw demand")
    df["First Difference"] = df["demand"] - df["demand"].shift(1)
    df["Seasonal First Difference"] = df["demand"] - df["demand"].shift(12)
    adfuller_test(df["Seasonal First Difference"].dropna(), "seasonal first difference")

    df["Seasonal First Difference"].plot(figsize=(12, 5), title="Seasonal First Difference")
    plt.tight_layout()
    plt.savefig(RESULTS / "02_seasonal_diff.png", dpi=120)
    plt.close()

    # Step 4: ACF / PACF for order selection
    autocorrelation_plot(df["demand"])
    plt.title("Autocorrelation of demand")
    plt.tight_layout()
    plt.savefig(RESULTS / "03_autocorrelation.png", dpi=120)
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    plot_acf(df["Seasonal First Difference"].iloc[13:], lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    plot_pacf(df["Seasonal First Difference"].iloc[13:], lags=40, ax=ax2)
    plt.tight_layout()
    plt.savefig(RESULTS / "04_acf_pacf.png", dpi=120)
    plt.close()

    # Step 5: non-seasonal ARIMA(1,1,1)
    print("\nStep 5: Fit ARIMA(1,1,1)")
    arima_fit = ARIMA(df["demand"], order=(1, 1, 1)).fit()
    print(arima_fit.summary())

    df["arima_forecast"] = arima_fit.predict(start=90, end=103, dynamic=True)
    df[["demand", "arima_forecast"]].plot(figsize=(12, 6), title="ARIMA(1,1,1) in-sample forecast")
    plt.tight_layout()
    plt.savefig(RESULTS / "05_arima_insample.png", dpi=120)
    plt.close()

    # Step 6: SARIMA(1,1,1)(1,1,1,12)
    print("\nStep 6: Fit SARIMAX(1,1,1)(1,1,1,12)")
    sarima_fit = sm.tsa.statespace.SARIMAX(
        df["demand"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
    ).fit(disp=False)
    print(sarima_fit.summary())

    df["sarima_forecast"] = sarima_fit.predict(start=90, end=103, dynamic=True)
    df[["demand", "sarima_forecast"]].plot(figsize=(12, 6), title="SARIMA in-sample forecast")
    plt.tight_layout()
    plt.savefig(RESULTS / "06_sarima_insample.png", dpi=120)
    plt.close()

    # Step 7: 24-month out-of-sample forecast
    print("\nStep 7: 24-month forecast")
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, 25)]
    future_df = pd.DataFrame(index=future_dates, columns=df.columns)
    full = pd.concat([df, future_df])
    full["future_forecast"] = sarima_fit.predict(start=len(df), end=len(df) + 23, dynamic=True)

    full[["demand", "future_forecast"]].plot(figsize=(12, 6), title="SARIMA 24-month forecast")
    plt.tight_layout()
    plt.savefig(RESULTS / "07_future_forecast.png", dpi=120)
    plt.close()

    forecast_out = full.loc[future_dates, "future_forecast"].rename("forecast")
    forecast_out.to_csv(RESULTS / "future_forecast_24m.csv", header=True)
    print("\n24-month forecast (head):")
    print(forecast_out.head(12).round(1))
    print(f"\nAll outputs written to: {RESULTS}")


if __name__ == "__main__":
    main()
