import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

class MovingAverageForecastAgent:
    """Forecasting using moving averages."""
    def forecast(self, sales_data):
        if len(sales_data) < 6:
            return [max(0, round(np.mean(sales_data)))] * 6
        ma_forecast = pd.Series(sales_data).rolling(window=12, min_periods=6).mean().dropna().values[-1]
        return [max(0, round(ma_forecast))] * 6

class SlowMovingForecastAgent:
    """Forecasting for slow-moving materials using ARIMA."""
    def forecast(self, sales_data):
        if len(sales_data) < 12:
            return [max(0, round(np.mean(sales_data)))] * 6
        model = ARIMA(sales_data, order=(5, 1, 0))
        model_fit = model.fit()
        return [max(0, round(value)) for value in model_fit.forecast(steps=6)]

class SeasonalForecastAgent:
    """Forecasting for seasonal materials using Prophet."""
    def forecast(self, sales_data):
        if len(sales_data) < 5:
            return [max(0, round(np.mean(sales_data)))] * 6
        df = pd.DataFrame({"ds": pd.date_range(start="2024-01-01", periods=len(sales_data), freq="M"), "y": sales_data})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=6, freq="M")
        forecast_df = model.predict(future)
        return [max(0, round(value)) for value in forecast_df["yhat"].tail(6)]
