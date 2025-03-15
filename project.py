import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch Bitcoin price data from CoinGecko API
def fetch_bitcoin_data(days=365):
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        prices = data["prices"]
        
        # Convert timestamp to datetime and extract price
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        return df
    else:
        print("Error fetching data:", response.status_code)
        return None

# Fetch the data
df = fetch_bitcoin_data()

if df is not None:
    # Plot the original Bitcoin price data
    plt.figure(figsize=(10, 5))
    plt.plot(df["price"], label="Bitcoin Price (USD)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Trend")
    plt.legend()
    plt.show()

    # Fit ARIMA model (p, d, q) parameters
    p, d, q = 2, 1, 2  # You can tune these parameters based on ACF/PACF analysis
    model = ARIMA(df["price"], order=(p, d, q))
    fitted_model = model.fit()

    # Forecast the next 10 days
    forecast_steps = 10
    forecast = fitted_model.forecast(steps=forecast_steps)

    # Create a date range for the forecast
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq="D")[1:]

    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["price"], label="Actual Prices", color="blue")
    plt.plot(forecast_dates, forecast, label="Predicted Prices", color="red", linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Bitcoin Price Prediction using ARIMA")
    plt.legend()
    plt.show()

    # Print forecasted prices
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Predicted Price": forecast})
    print(forecast_df)
else:
    print("Data retrieval failed.")
    