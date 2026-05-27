import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model

TICKER = "AAPL"
PERIOD = "3y"
WINDOW_SIZE = 30
MODEL_DIR = "models"


def load_returns(ticker: str, period: str) -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")
    close = data["Close"].dropna()
    returns = close.pct_change().dropna()
    return returns


def main():
    model_path = os.path.join(MODEL_DIR, f"{TICKER}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{TICKER}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Не найдена модель: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Не найден scaler: {scaler_path}")

    model = load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    returns = load_returns(TICKER, PERIOD)
    scaled = scaler.transform(returns.values.reshape(-1, 1))

    last_window = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    next_return_scaled = model.predict(last_window, verbose=0)
    next_return = scaler.inverse_transform(next_return_scaled)[0][0]

    print(f"Прогноз доходности следующего дня для {TICKER}: {next_return:.6f}")


if __name__ == "__main__":
    main()