# eda_modules/forecast.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_forecast(df, target_column="sales", steps=365):
    if "date" not in df.columns or target_column not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target_column])
    df = df.sort_values("date")
    df["t"] = np.arange(len(df))

    X = df[["t"]]
    y = df[target_column]

    model = LinearRegression()
    model.fit(X, y)

    future_t = np.arange(len(df), len(df) + steps).reshape(-1, 1)
    forecast = model.predict(future_t)
    forecast_dates = pd.date_range(df["date"].max(), periods=steps+1, freq="D")[1:]

    forecast_df = pd.DataFrame({"date": forecast_dates, f"forecast_{target_column}": forecast})

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(df["date"], df[target_column], label="Actual")
    ax.plot(forecast_df["date"], forecast_df[f"forecast_{target_column}"], label="Forecast", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_column.capitalize())
    ax.set_title("Forecast vs. Actual")
    ax.legend()

    return fig
