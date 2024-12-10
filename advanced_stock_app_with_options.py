
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
    # Check if the data has enough rows
    if len(data) < window:
        st.error(f"Not enough data to calculate Bollinger Bands. Minimum required: {window} rows.")
        return data

    # Check if 'Close' column exists
    if 'Close' not in data.columns:
        st.error("Missing 'Close' column in the dataset.")
        return data

    # Calculate Bollinger Bands
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
    if len(data) < 2:
        st.error("Not enough data for price predictions.")
        return None
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
        if not hist.empty:
            recent_gain = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0] * 100
            data[ticker] = recent_gain
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    return sorted_data[:5]

# Recommend Call/Put Options
def recommend_options(data):
    if len(data) < 2:
        st.error("Not enough data to calculate options recommendations.")
        return "No Recommendations"

    last_close = data['Close'].iloc[-1]
    next_prediction = predict_prices(data, days=1)[0] if predict_prices(data, days=1) is not None else last_close

    if next_prediction > last_close:
        return f"Call Option Recommended: Current Price = {last_close:.2f}, Predicted Price = {next_prediction:.2f}"
    else:
        return f"Put Option Recommended: Current Price = {last_close:.2f}, Predicted Price = {next_prediction:.2f}"

# Streamlit App
st.title("Advanced Stock Market Analysis App")

# Sidebar Inputs
st.sidebar.header("Stock Analysis Options")
top_trades = fetch_top_trades()
top_tickers = [ticker for ticker, _ in top_trades]
selected_stock = st.sidebar.selectbox("Select a Stock from Top Performers", top_tickers)
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
days_to_predict = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)

# Fetch and analyze data
if st.sidebar.button("Analyze"):
    st.subheader(f"Analysis for {selected_stock}")
    stock_data = fetch_stock_data(selected_stock, start_date, end_date)

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
        if predictions is not None:
            st.write(f"Predicted Prices for the next {days_to_predict} days:")
            st.write(predictions)

        # Recommend Options
        recommendation = recommend_options(stock_data)
        st.write("Options Recommendation:")
        st.success(recommendation)

# Display top 5 trades
st.sidebar.header("Top 5 Trades")
if st.sidebar.button("Get Top Trades"):
    st.write("Top 5 Trades for Maximum Returns:")
    for ticker, gain in top_trades:
        st.write(f"{ticker}: {gain:.2f}%")
