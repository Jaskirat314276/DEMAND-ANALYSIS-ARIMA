"""Streamlit dashboard for the SARIMA champagne-demand forecaster."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from pandas.tseries.offsets import DateOffset
from statsmodels.tsa.stattools import adfuller

CSV = Path(__file__).parent / "perrin-freres-monthly-champagne-.csv"

st.set_page_config(page_title="Champagne Demand Forecaster", layout="wide")
st.title("Perrin Frères Champagne Demand — SARIMA Forecaster")
st.caption("Monthly demand 1964–1972 with a Seasonal ARIMA model.")


@st.cache_data
def load_data():
    df = pd.read_csv(CSV)
    df.columns = ["Month", "demand"]
    df = df.dropna().reset_index(drop=True)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")
    df.index.freq = "MS"
    return df


@st.cache_resource
def fit_model(p, d, q, P, D, Q, s):
    df = load_data()
    model = sm.tsa.statespace.SARIMAX(
        df["demand"],
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
    )
    return model.fit(disp=False)


df = load_data()

with st.sidebar:
    st.header("Model parameters")
    st.markdown("**Non-seasonal (p, d, q)**")
    p = st.slider("p (AR)", 0, 3, 1)
    d = st.slider("d (diff)", 0, 2, 1)
    q = st.slider("q (MA)", 0, 3, 1)
    st.markdown("**Seasonal (P, D, Q, s)**")
    P = st.slider("P", 0, 2, 1)
    D = st.slider("D", 0, 2, 1)
    Q = st.slider("Q", 0, 2, 1)
    s = st.number_input("s (period)", 1, 24, 12)
    st.markdown("---")
    horizon = st.slider("Forecast horizon (months)", 1, 36, 24)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Historical demand")
    fig, ax = plt.subplots(figsize=(10, 4))
    df["demand"].plot(ax=ax)
    ax.set_ylabel("demand")
    st.pyplot(fig)

with col2:
    st.subheader("Stationarity (ADF)")
    raw_p = adfuller(df["demand"])[1]
    seas_p = adfuller((df["demand"] - df["demand"].shift(12)).dropna())[1]
    st.metric("Raw series p-value", f"{raw_p:.3f}",
              delta="non-stationary" if raw_p > 0.05 else "stationary",
              delta_color="inverse")
    st.metric("Seasonal diff p-value", f"{seas_p:.2e}",
              delta="stationary" if seas_p <= 0.05 else "non-stationary")

st.markdown("---")
st.subheader(f"SARIMA({p},{d},{q})({P},{D},{Q},{s}) forecast")

with st.spinner("Fitting model..."):
    fit = fit_model(p, d, q, P, D, Q, s)

future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, horizon + 1)]
forecast = fit.get_forecast(steps=horizon)
mean = forecast.predicted_mean
mean.index = future_dates
ci = forecast.conf_int()
ci.index = future_dates

fig, ax = plt.subplots(figsize=(12, 5))
df["demand"].plot(ax=ax, label="historical")
mean.plot(ax=ax, label="forecast", color="orange")
ax.fill_between(future_dates, ci.iloc[:, 0], ci.iloc[:, 1], color="orange", alpha=0.2,
                label="95% CI")
ax.legend()
ax.set_ylabel("demand")
st.pyplot(fig)

c1, c2, c3 = st.columns(3)
c1.metric("AIC", f"{fit.aic:.1f}")
c2.metric("BIC", f"{fit.bic:.1f}")
c3.metric("Log-likelihood", f"{fit.llf:.1f}")

st.subheader("Forecast table")
out = pd.DataFrame({
    "forecast": mean.round(1),
    "lower_95": ci.iloc[:, 0].round(1),
    "upper_95": ci.iloc[:, 1].round(1),
})
st.dataframe(out, use_container_width=True)
st.download_button("Download forecast as CSV", out.to_csv().encode(),
                   file_name="forecast.csv", mime="text/csv")

with st.expander("Model summary"):
    st.text(str(fit.summary()))
