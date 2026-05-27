import os
import pickle
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
WINDOW_SIZE = 30
LSTM_DATA_PERIOD = "3y"


def load_returns(ticker: str, period: str = LSTM_DATA_PERIOD) -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if data.empty or "Close" not in data.columns:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")

    close = data["Close"].dropna()
    returns = close.pct_change().dropna()

    if len(returns) < WINDOW_SIZE:
        raise ValueError(f"Недостаточно данных для прогноза {ticker}")

    return returns


def predict_next_return(ticker: str) -> float:
    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель для {ticker} не найдена: {model_path}")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler для {ticker} не найден: {scaler_path}")

    model = load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    returns = load_returns(ticker)
    scaled = scaler.transform(returns.values.reshape(-1, 1))

    last_window = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    pred_scaled = model.predict(last_window, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    return float(pred)