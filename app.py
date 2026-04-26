import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Oil Price Forecasting Dashboard")

st.write("""
This dashboard analyzes crude oil price trends using moving averages
and a simple forecasting method.
""")

uploaded_file = st.file_uploader("Upload oil price CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Expected columns: Date, Price
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    df["7-day MA"] = df["Price"].rolling(window=7).mean()
    df["30-day MA"] = df["Price"].rolling(window=30).mean()

    st.subheader("Oil Price Trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Date"], df["Price"], label="Oil Price")
    ax.plot(df["Date"], df["7-day MA"], label="7-day Moving Average")
    ax.plot(df["Date"], df["30-day MA"], label="30-day Moving Average")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title("Crude Oil Price Trend")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    st.subheader("Simple Forecast")

    last_price = df["Price"].iloc[-1]
    forecast_days = st.slider("Select forecast days", 5, 60, 30)

    forecast_dates = pd.date_range(
        start=df["Date"].iloc[-1],
        periods=forecast_days + 1,
        freq="D"
    )[1:]

    forecast_price = df["Price"].tail(30).mean()
    forecast_values = np.full(forecast_days, forecast_price)

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast Price": forecast_values
    })

    st.dataframe(forecast_df)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(df["Date"], df["Price"], label="Historical Price")
    ax2.plot(forecast_df["Date"], forecast_df["Forecast Price"], label="Forecast Price")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.set_title("Simple Moving Average Forecast")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file with columns: Date and Price.")
