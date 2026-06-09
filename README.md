<div align="center">

# Champagne Demand Forecaster

**Seasonal ARIMA on 9 years of Perrin Frères monthly champagne sales — as a one-shot script *and* an interactive Streamlit dashboard.**

[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)](#)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14%2B-4C72B0)](#)
[![License](https://img.shields.io/badge/license-MIT-22c55e)](#)

</div>

---

## Overview

Champagne sales explode every December and crash every August. A plain ARIMA would smooth that pattern away, so this project uses **Seasonal ARIMA (SARIMA)** to capture both the trend and the yearly cycle, then projects 24 months ahead.

You get two ways to use it:

- **`app.py`** — a dark-themed Streamlit dashboard with live order tuning, KPI cards, a Plotly forecast chart with 95% confidence bands, and CSV export.
- **`run_sarima.py`** — a non-interactive end-to-end pipeline that writes diagnostic plots + a forecast CSV to `./results/`.

## Demo at a glance

| | |
|---|---|
| **Dataset** | Perrin Frères monthly champagne, 1964–1972 (105 observations) |
| **Model** | `SARIMAX(order=(1,1,1), seasonal_order=(1,1,1,12))` |
| **Default horizon** | 24 months |
| **Headline metric** | AIC ≈ **1487** vs. ARIMA(1,1,1) baseline of ≈ 1912 |
| **Seasonal ADF p-value** | ≈ 2 × 10⁻¹¹ (stationary after seasonal differencing) |

## Repository layout

```
.
├─ app.py                              # Streamlit dashboard (modern dark UI)
├─ run_sarima.py                       # Reproducible end-to-end pipeline
├─ perrin-freres-monthly-champagne-.csv  # Source dataset
├─ Untitled.ipynb                      # Original exploratory notebook
├─ requirements.txt
└─ results/                            # Plots + forecast CSV (created by run_sarima.py)
   ├─ 01_raw_series.png
   ├─ 02_seasonal_diff.png
   ├─ 03_autocorrelation.png
   ├─ 04_acf_pacf.png
   ├─ 05_arima_insample.png
   ├─ 06_sarima_insample.png
   ├─ 07_future_forecast.png
   └─ future_forecast_24m.csv
```

## Quick start

```bash
git clone https://github.com/Jaskirat314276/DEMAND-ANALYSIS-ARIMA.git
cd DEMAND-ANALYSIS-ARIMA

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Launch the dashboard

```bash
streamlit run app.py
```

Then open `http://localhost:8501`. Use the sidebar to tweak `(p, d, q)`, `(P, D, Q, s)`, and the forecast horizon — the chart, KPI cards, and forecast table refresh instantly.

### Run the headless pipeline

```bash
python run_sarima.py
```

Outputs land in `./results/`:

- ADF stationarity tests printed to stdout
- ACF / PACF diagnostic plots
- In-sample fit plots for ARIMA and SARIMA
- 24-month out-of-sample forecast plot
- `future_forecast_24m.csv`

## What's in the dashboard

- **KPI strip** — sample size, mean demand, peak month, and the ADF p-value on the seasonally-differenced series (turns green when stationary).
- **Forecast chart** — historical series in blue, point forecast in orange, 95% confidence band shaded. Hover for per-month values.
- **Model fit row** — chosen order, AIC, BIC, log-likelihood. Toggle sliders and watch AIC respond.
- **Forecast table** — point + lower/upper 95% bounds, downloadable as CSV.
- **Full model summary** — collapsible `statsmodels` summary with coefficients, p-values, and Ljung-Box diagnostics.
- **Guide tab** — built-in tutorial covering parameters, tuning tips, and how to adapt the app to your own dataset.

## SARIMA parameters at a glance

| Param | Meaning | Sensible range |
|-------|---------|----------------|
| **p** | AR — how many past values feed the prediction | 0–2 |
| **d** | Differencing — removes trend | 1 |
| **q** | MA — how many past errors feed the prediction | 0–2 |
| **P** | Seasonal AR — past values from the same season last year | 0–1 |
| **D** | Seasonal differencing — removes yearly seasonality | 1 |
| **Q** | Seasonal MA — past errors from the same season | 0–1 |
| **s** | Length of the seasonal cycle (12 = monthly with yearly pattern) | 12 |

Defaults `(1,1,1)(1,1,1,12)` are what the original notebook converged on.

## Methodology

1. **Clean** — parse the date column, drop the trailing footer rows in the source CSV, set a monthly `DatetimeIndex` with `freq="MS"`.
2. **Inspect stationarity** — Augmented Dickey–Fuller on the raw series fails; after seasonal differencing (lag 12), p ≈ 2 × 10⁻¹¹.
3. **Pick orders** — ACF / PACF on the seasonally-differenced series suggest one AR and one MA term, mirrored at the seasonal lag.
4. **Fit baseline** — `ARIMA(1,1,1)`, AIC ≈ 1912.
5. **Fit SARIMA** — `SARIMAX(1,1,1)(1,1,1,12)`, AIC ≈ 1487. Large drop confirms seasonality is the dominant signal.
6. **Forecast** — 24 months ahead with 95% confidence bands.

## Adapting to your own data

Replace `perrin-freres-monthly-champagne-.csv` with any two-column CSV (date + numeric value) and adjust `s` to your seasonal period:

| Frequency | `s` |
|-----------|-----|
| Monthly with a yearly pattern | 12 |
| Daily with a weekly pattern | 7 |
| Hourly with a daily pattern | 24 |
| Quarterly with a yearly pattern | 4 |

The Streamlit cache picks up the new file automatically on next run.

## Tech stack

`Python 3.9+` · `streamlit` · `statsmodels` · `pandas` · `numpy` · `plotly` · `matplotlib`

## Roadmap

- [ ] **SARIMAX** with exogenous regressors (promotions, holidays, pricing)
- [ ] **Walk-forward validation** with MAE / MAPE on rolling windows
- [ ] **Hybrid SARIMA + LSTM** — neural net on the residuals
- [ ] **Auto-order search** (grid or `pmdarima.auto_arima`) toggle in the sidebar

## License

MIT.
