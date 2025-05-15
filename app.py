import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Streamlit Config ---
st.set_page_config(page_title="Stock Forecasting", layout="wide")

# --- Theme Switch ---
st.sidebar.title("Navigation")
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
model = st.sidebar.selectbox("Choose Model", ["ARIMA", "Prophet", "LSTM"])
uploaded_file = st.sidebar.file_uploader("Upload Stock Data (CSV)", type="csv")

# --- Theme Styling ---
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0e1117;
            color: white;
        }
        .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Data Loading & Preprocessing ---
def load_and_preprocess(df):
    df.columns = [col.strip().lower() for col in df.columns]
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    df = df.sort_index()
    return df[['close']]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Stock_data.csv")  # default file

df = load_and_preprocess(df)

# --- Forecasting Models ---

def run_arima(df):
    series = df['close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    forecast_index = pd.date_range(start=series.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=forecast_index)

def run_prophet(df):
    data = df.reset_index()[['date', 'close']]
    data.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    forecast_result = forecast[['ds', 'yhat']].set_index('ds').tail(30)
    return forecast_result['yhat']

def run_lstm(df):
    data = df['close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(data_scaled)):
        X.append(data_scaled[i-60:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    forecast_input = data_scaled[-60:].reshape(1, 60, 1)
    forecast = []

    for _ in range(30):
        pred = model.predict(forecast_input)[0][0]
        forecast.append(pred)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[pred]]], axis=1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_index = pd.date_range(start=df.index[-1], periods=30, freq='B')
    return pd.Series(forecast, index=forecast_index)

# --- App Layout ---
st.title("Stock Market Forecasting Dashboard")

tab1, tab2 = st.tabs(["**Data Overview**", "**Model Forecasting**"])

with tab1:
    st.subheader("Stock Closing Price")
    st.line_chart(df['close'])

    st.subheader("Raw Data")
    st.dataframe(df.tail(10), use_container_width=True)

with tab2:
    st.subheader(f"{model} Forecast")

    if model == "ARIMA":
        with st.expander("What is ARIMA?"):
            st.write("ARIMA is a statistical model combining autoregression, differencing, and moving average.")
        forecast = run_arima(df)

    elif model == "Prophet":
        with st.expander("What is Prophet?"):
            st.write("Prophet is a model developed by Facebook designed to handle seasonality and trend changes.")
        forecast = run_prophet(df)

    elif model == "LSTM":
        with st.expander("What is LSTM?"):
            st.write("LSTM (Long Short-Term Memory) is a deep learning model suited for sequential prediction.")
        forecast = run_lstm(df)

    # Plot forecast
    st.line_chart(pd.concat([df['close'].iloc[-30:], forecast], axis=0))

    st.subheader("Forecast Table")
    st.dataframe(forecast.to_frame(name="Forecast"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Developed by Mayakshif | Data Analytics Project**")