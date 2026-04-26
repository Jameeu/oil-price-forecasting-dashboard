from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Oil Forecasting Lab",
    page_icon="",
    layout="wide",
)


@dataclass(frozen=True)
class ForecastResult:
    history: pd.DataFrame
    forecast: pd.DataFrame
    model_name: str
    daily_volatility: float
    annualized_volatility: float
    last_price: float
    expected_final_price: float


def make_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=520)

    prices = [76.0]
    long_run_log_price = math.log(82.0)
    mean_reversion_speed = 0.035
    daily_vol = 0.018

    for i in range(1, len(dates)):
        seasonal = 0.0018 * math.sin(i / 28)
        shock = rng.normal(0, daily_vol)
        previous_log = math.log(prices[-1])
        next_log = (
            previous_log
            + mean_reversion_speed * (long_run_log_price - previous_log)
            + seasonal
            + shock
        )
        prices.append(float(np.clip(math.exp(next_log), 42, 128)))

    return pd.DataFrame({"date": dates, "price": prices})


def clean_price_data(raw: pd.DataFrame) -> pd.DataFrame:
    lower_to_original = {column.lower().strip(): column for column in raw.columns}

    date_column = lower_to_original.get("date")
    price_column = lower_to_original.get("price")

    if date_column is None or price_column is None:
        raise ValueError("CSV must contain columns named date and price.")

    df = raw[[date_column, price_column]].copy()
    df.columns = ["date", "price"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna().sort_values("date")
    df = df[df["price"] > 0]
    df = df.drop_duplicates(subset="date", keep="last")

    if len(df) < 40:
        raise ValueError("Need at least 40 valid price observations.")

    return df.reset_index(drop=True)


def calculate_log_returns(df: pd.DataFrame) -> pd.Series:
    return np.log(df["price"] / df["price"].shift(1)).dropna()


def forecast_random_walk(df: pd.DataFrame, horizon: int, confidence_width: float) -> ForecastResult:
    returns = calculate_log_returns(df)
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    last_price = float(df["price"].iloc[-1])

    steps = np.arange(1, horizon + 1)
    expected_log = math.log(last_price) + mu * steps
    uncertainty = confidence_width * sigma * np.sqrt(steps)

    forecast = build_forecast_frame(df, expected_log, uncertainty)

    return ForecastResult(
        history=df,
        forecast=forecast,
        model_name="Random walk with drift",
        daily_volatility=sigma,
        annualized_volatility=sigma * math.sqrt(252),
        last_price=last_price,
        expected_final_price=float(forecast["forecast"].iloc[-1]),
    )


def forecast_moving_average(df: pd.DataFrame, horizon: int, window: int, confidence_width: float) -> ForecastResult:
    returns = calculate_log_returns(df)
    recent_returns = returns.tail(window)
    mu = float(recent_returns.mean())
    sigma = float(recent_returns.std(ddof=1))
    last_price = float(df["price"].iloc[-1])

    steps = np.arange(1, horizon + 1)
    expected_log = math.log(last_price) + mu * steps
    uncertainty = confidence_width * sigma * np.sqrt(steps)

    forecast = build_forecast_frame(df, expected_log, uncertainty)

    return ForecastResult(
        history=df,
        forecast=forecast,
        model_name=f"Moving-average drift ({window} days)",
        daily_volatility=sigma,
        annualized_volatility=sigma * math.sqrt(252),
        last_price=last_price,
        expected_final_price=float(forecast["forecast"].iloc[-1]),
    )


def forecast_mean_reversion(
    df: pd.DataFrame,
    horizon: int,
    half_life: int,
    long_run_price: float,
    confidence_width: float,
) -> ForecastResult:
    returns = calculate_log_returns(df)
    sigma = float(returns.std(ddof=1))
    last_price = float(df["price"].iloc[-1])

    kappa = math.log(2) / half_life
    long_run_log = math.log(long_run_price)
    current_log = math.log(last_price)

    steps = np.arange(1, horizon + 1)
    expected_log = long_run_log + (current_log - long_run_log) * np.exp(-kappa * steps)
    uncertainty = confidence_width * sigma * np.sqrt((1 - np.exp(-2 * kappa * steps)) / (2 * kappa))

    forecast = build_forecast_frame(df, expected_log, uncertainty)

    return ForecastResult(
        history=df,
        forecast=forecast,
        model_name=f"Mean reversion, {half_life}-day half-life",
        daily_volatility=sigma,
        annualized_volatility=sigma * math.sqrt(252),
        last_price=last_price,
        expected_final_price=float(forecast["forecast"].iloc[-1]),
    )


def build_forecast_frame(df: pd.DataFrame, expected_log: np.ndarray, uncertainty: np.ndarray) -> pd.DataFrame:
    last_date = pd.Timestamp(df["date"].iloc[-1])
    forecast_dates = pd.bdate_range(start=last_date + pd.offsets.BDay(1), periods=len(expected_log))

    return pd.DataFrame(
        {
            "date": forecast_dates,
            "forecast": np.exp(expected_log),
            "lower": np.exp(expected_log - uncertainty),
            "upper": np.exp(expected_log + uncertainty),
        }
    )


def build_price_chart(result: ForecastResult) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=result.history["date"],
            y=result.history["price"],
            mode="lines",
            name="Historical price",
            line=dict(color="#1f2937", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["upper"],
            mode="lines",
            name="Upper band",
            line=dict(color="rgba(14, 165, 233, 0.0)"),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["lower"],
            mode="lines",
            name="Forecast range",
            fill="tonexty",
            fillcolor="rgba(14, 165, 233, 0.18)",
            line=dict(color="rgba(14, 165, 233, 0.0)"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.forecast["date"],
            y=result.forecast["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color="#0284c7", width=3),
        )
    )

    fig.update_layout(
        height=520,
        margin=dict(l=24, r=24, t=24, b=24),
        hovermode="x unified",
        yaxis_title="USD per barrel",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_return_chart(df: pd.DataFrame) -> go.Figure:
    returns = calculate_log_returns(df)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=100 * returns,
            nbinsx=36,
            marker_color="#475569",
            name="Daily log returns",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=24, r=24, t=24, b=24),
        xaxis_title="Daily log return (%)",
        yaxis_title="Frequency",
        showlegend=False,
    )
    return fig


def market_regime_text(result: ForecastResult) -> str:
    expected_change = result.expected_final_price / result.last_price - 1
    vol = result.annualized_volatility

    if expected_change > 0.08:
        direction = "bullish"
    elif expected_change < -0.08:
        direction = "bearish"
    else:
        direction = "range-bound"

    if vol > 0.45:
        risk = "very high"
    elif vol > 0.30:
        risk = "high"
    elif vol > 0.18:
        risk = "moderate"
    else:
        risk = "low"

    return (
        f"The model implies a {direction} price path over the selected horizon, "
        f"with {risk} realized volatility based on the input data."
    )


def load_data_from_sidebar() -> pd.DataFrame:
    uploaded_file = st.sidebar.file_uploader("Upload oil price CSV", type=["csv"])

    if uploaded_file is None:
        return make_sample_data()

    return clean_price_data(pd.read_csv(uploaded_file))


def app() -> None:
    st.title("Oil Forecasting Lab")
    st.caption("A practical crude oil forecasting workspace for price paths, volatility, and risk bands.")

    st.sidebar.header("Data")
    try:
        df = load_data_from_sidebar()
    except Exception as exc:
        st.sidebar.error(str(exc))
        st.stop()

    st.sidebar.header("Forecast")
    model = st.sidebar.selectbox(
        "Model",
        ["Mean reversion", "Random walk with drift", "Moving-average drift"],
    )
    horizon = st.sidebar.slider("Forecast horizon, business days", 10, 260, 90, 5)
    confidence_width = st.sidebar.slider("Uncertainty band width", 0.5, 3.0, 1.65, 0.05)

    if model == "Mean reversion":
        long_run_price = st.sidebar.number_input(
            "Long-run price, USD/bbl",
            min_value=1.0,
            max_value=300.0,
            value=float(df["price"].tail(120).mean()),
            step=1.0,
        )
        half_life = st.sidebar.slider("Mean-reversion half-life, days", 10, 260, 75, 5)
        result = forecast_mean_reversion(df, horizon, half_life, long_run_price, confidence_width)
    elif model == "Moving-average drift":
        window = st.sidebar.slider("Return lookback window, days", 10, 252, 60, 5)
        result = forecast_moving_average(df, horizon, window, confidence_width)
    else:
        result = forecast_random_walk(df, horizon, confidence_width)

    latest_date = pd.Timestamp(df["date"].iloc[-1]).date()
    expected_change = result.expected_final_price / result.last_price - 1

    metric_columns = st.columns(4)
    metric_columns[0].metric("Latest price", f"${result.last_price:,.2f}", str(latest_date))
    metric_columns[1].metric("Forecast final", f"${result.expected_final_price:,.2f}", f"{expected_change:+.1%}")
    metric_columns[2].metric("Annualized volatility", f"{result.annualized_volatility:.1%}")
    metric_columns[3].metric("Observations", f"{len(df):,}")

    st.plotly_chart(build_price_chart(result), use_container_width=True)

    lower_final = float(result.forecast["lower"].iloc[-1])
    upper_final = float(result.forecast["upper"].iloc[-1])
    st.info(
        f"{market_regime_text(result)} Final-day range: "
        f"${lower_final:,.2f} to ${upper_final:,.2f}. Model: {result.model_name}."
    )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Return Distribution")
        st.plotly_chart(build_return_chart(df), use_container_width=True)

    with right:
        st.subheader("Forecast Table")
        display = result.forecast.copy()
        display["date"] = display["date"].dt.date
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "forecast": st.column_config.NumberColumn("forecast", format="$%.2f"),
                "lower": st.column_config.NumberColumn("lower", format="$%.2f"),
                "upper": st.column_config.NumberColumn("upper", format="$%.2f"),
            },
        )

        csv = result.forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download forecast CSV",
            data=csv,
            file_name="oil_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Model notes"):
        st.write(
            """
            These forecasts are educational models, not trading signals.

            Random walk with drift uses the historical average log return.
            Moving-average drift uses recent average log return.
            Mean reversion assumes log prices are pulled toward a chosen long-run level.

            For professional oil forecasting, combine price history with inventories,
            futures curves, OPEC policy, refinery utilization, interest rates, FX,
            geopolitical risk, and macro demand indicators.
            """
        )


if __name__ == "__main__":
    app()

