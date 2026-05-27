import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

TICKER = "AAPL"
PERIOD = "3y"
WINDOW_SIZE = 30

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_returns(ticker: str, period: str) -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")
    close = data["Close"].dropna()
    returns = close.pct_change().dropna()
    return returns


def create_sequences(values: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def build_model(window_size: int) -> Sequential:
    model = Sequential([
        Input(shape=(window_size, 1)),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    print(f"Загрузка данных для {TICKER}...")
    returns = load_returns(TICKER, PERIOD)

    split_idx_raw = int(len(returns) * 0.8)

    train_returns = returns.iloc[:split_idx_raw]
    test_returns = returns.iloc[split_idx_raw - WINDOW_SIZE:]

    scaler = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler.fit_transform(
        train_returns.values.reshape(-1, 1)
    )

    test_scaled = scaler.transform(
        test_returns.values.reshape(-1, 1)
    )

    X_train, y_train = create_sequences(train_scaled, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_scaled, WINDOW_SIZE)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_model(WINDOW_SIZE)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Обучение модели...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    print("Прогнозирование...")
    pred_scaled = model.predict(X_test, verbose=0)

    pred = scaler.inverse_transform(pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Реальная доходность")
    plt.plot(pred, label="Прогноз LSTM")
    plt.title(f"LSTM прогноз доходности для {TICKER}")
    plt.legend()
    plt.grid(True)
    plt.show()

    last_window = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    next_return_scaled = model.predict(last_window, verbose=0)
    next_return = scaler.inverse_transform(next_return_scaled)[0][0]

    print(f"Прогноз доходности следующего дня для {TICKER}: {next_return:.6f}")

    model_path = os.path.join(MODEL_DIR, f"{TICKER}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{TICKER}_scaler.pkl")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Модель сохранена: {model_path}")
    print(f"Scaler сохранён: {scaler_path}")


if __name__ == "__main__":
    main()