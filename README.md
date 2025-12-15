# ARIMA & Seasonal ARIMA — Time Series Forecasting

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#)

> This repository walks through a practical implementation of **ARIMA** and **Seasonal ARIMA (SARIMA)** models to forecast monthly champagne demand (1964–1972). It covers the complete workflow—from preparing the dataset to choosing the right model and generating reliable 24‑month forecasts.


# Highlights

* Step-by-step walkthrough of ARIMA and SARIMA modeling.
* Hands-on time-series techniques: **Dickey–Fuller test**, **seasonal differencing**, **ACF/PACF interpretation**.
* Clear, reproducible structure: data cleaning → diagnostics → model building → forecasting.
* Simple, readable notebooks designed for learning and real-world application.


## 🚀 Purpose of this repo

Many tutorials skip essential reasoning steps. Here, the focus is on understanding *why* each transformation or parameter is used. This means:

* Knowing when and why to difference a time series.
* Reading ACF/PACF plots with confidence when selecting AR and MA lags.
* Understanding the effect of seasonality (12-month cycles) on model choice.


# Repository structure

```
ARIMA-And-Seasonal-ARIMA/
├─ data/                     # raw and cleaned CSVs
├─ notebooks/                # Jupyter notebooks walkthrough
│  ├─ 01_data_preparation.ipynb
│  ├─ 02_stationarity_and_diff.ipynb
│  └─ 03_sarima_modeling.ipynb
├─ scripts/                  # reusable scripts (train, forecast, utils)
├─ results/                  # model outputs and plots
├─ README.md
└─ requirements.txt
```

# Tech stack

* Python 3.8+
* pandas, numpy
* matplotlib, seaborn
* statsmodels
* scikit-learn
* Jupyter Notebook


# Getting started

Clone the repository and install requirements:

```bash
git clone https://github.com/Jaskirat314276/DEMAND-ANALYSIS-ARIMA.git
cd ARIMA-And-Seasonal-ARIMA
python -m venv venv
source venv/bin/activate     # macOS / Linux
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

Run the notebooks one by one:

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

To run the script version (if included):

```bash
python scripts/train_sarima.py --data data/champagne_demand.csv --forecast 24
```


# What you'll learn

* How to check stationarity using the **ADF test**.
* When to apply **seasonal differencing**.
* How to read **ACF/PACF** plots to choose AR and MA orders.
* How to fit and tune **SARIMA** models.
* How to create clean visualizations for time series trends and forecasts.


# Sample model

```
SARIMA(order=(1,1,1), seasonal_order=(1,1,1,12))
```

This configuration worked well on the champagne dataset, capturing clear yearly patterns and producing a 24‑month forecast. Generated plots and outputs can be found in the `results` folder.


# Reproducibility

* Ensure the date column uses a proper `DatetimeIndex`.
* Avoid shuffling when splitting time‑series data.
* When needed, fix random seeds and log environment versions.


# Ideas for improvement

Here are some natural next steps if you want to extend the project:

* Try LSTM/GRU models for learning nonlinear patterns.
* Build a Hybrid SARIMA + LSTM** model by forecasting with SARIMA and modeling residuals with a neural network.
* Add **exogenous variables** (promotions, holidays, pricing) via SARIMAX.
* Implement **walk-forward validation** for a more robust evaluation.

