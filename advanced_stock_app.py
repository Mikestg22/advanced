
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    data['Rolling_Mean'] = data['Close'].rolling(window=window).mean()
    data['Upper_Band'] = data['Rolling_Mean'] + (data['Close'].rolling(window=window).std() * 2)
    data['Lower_Band'] = data['Rolling_Mean'] - (data['Close'].rolling(window=window).std() * 2)
    return data

# Calculate Exponential Moving Average
def calculate_ema(data, span=20):
    data[f'EMA_{span}'] = data['Close'].ewm(span=span, adjust=False).mean()
    return data

# Predict Future Prices
def predict_prices(data, days=7):
    data['Days'] = np.arange(len(data))
    X = data[['Days']].values
    y = data['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    future_days = np.arange(len(data), len(data) + days).reshape(-1, 1)
    predictions = model.predict(future_days)
    return predictions

# Fetch top 5 trades
def fetch_top_trades():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        recent_gain = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100
        data[ticker] = recent_gain
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return sorted_data[:5]

# Streamlit App
st.title("Advanced Stock Market App")

# Sidebar Inputs
st.sidebar.header("Stock Analysis Options")
selected_stocks = st.sidebar.text_input("Enter Stock Tickers (comma-separated)", "AAPL,MSFT")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
days_to_predict = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)

# Fetch and analyze data
if st.sidebar.button("Analyze"):
    tickers = [ticker.strip() for ticker in selected_stocks.split(",")]
    for ticker in tickers:
        st.subheader(f"Analysis for {ticker}")
        stock_data = fetch_stock_data(ticker, start_date, end_date)

        if stock_data is not None and not stock_data.empty:
            # Calculate indicators
            stock_data = calculate_bollinger_bands(stock_data)
            stock_data = calculate_ema(stock_data)

            # Display data
            st.write("Recent Data")
            st.dataframe(stock_data.tail(10))

            # Plot Bollinger Bands
            st.write("Bollinger Bands")
            fig, ax = plt.subplots()
            ax.plot(stock_data.index, stock_data['Close'], label="Close Price")
            ax.plot(stock_data.index, stock_data['Upper_Band'], label="Upper Band", linestyle="--")
            ax.plot(stock_data.index, stock_data['Lower_Band'], label="Lower Band", linestyle="--")
            ax.fill_between(stock_data.index, stock_data['Upper_Band'], stock_data['Lower_Band'], alpha=0.1)
            ax.legend()
            st.pyplot(fig)

            # Plot EMA
            st.write("Exponential Moving Average (EMA)")
            fig, ax = plt.subplots()
            ax.plot(stock_data.index, stock_data['Close'], label="Close Price")
            ax.plot(stock_data.index, stock_data[f'EMA_20'], label="EMA (20)", linestyle="--")
            ax.legend()
            st.pyplot(fig)

            # Predict Prices
            predictions = predict_prices(stock_data, days=days_to_predict)
            st.write(f"Predicted Prices for the next {days_to_predict} days:")
            st.write(predictions)

# Display top 5 trades
st.sidebar.header("Top 5 Trades")
if st.sidebar.button("Get Top Trades"):
    top_trades = fetch_top_trades()
    st.write("Top 5 Trades for Maximum Returns:")
    for ticker, gain in top_trades:
        st.write(f"{ticker}: {gain:.2f}%")
