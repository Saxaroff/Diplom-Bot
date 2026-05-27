import os
import pickle
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# =========================
# НАСТРОЙКИ
# =========================
TICKERS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "GOOGL", "META", "NFLX", "AMD", "INTC",
    "JPM", "BAC", "GS", "WFC", "V",
    "MA", "XOM", "CVX", "KO", "PEP",
    "JNJ", "PFE", "UNH", "DIS", "NKE",
    "MCD", "WMT", "COST", "ABNB", "UBER"
]

PERIOD = "10y"
WINDOW_SIZE = 30
EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2
MODEL_DIR = "models"
PLOTS_DIR = "plots"
RESULTS_FILE = "training_results.csv"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# =========================
# ЗАГРУЗКА ДАННЫХ
# =========================
def load_returns(ticker: str, period: str) -> pd.Series:
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if data.empty or "Close" not in data.columns:
        raise ValueError(f"Не удалось загрузить данные для {ticker}")

    close = data["Close"].dropna()
    returns = close.pct_change().dropna()

    if len(returns) < WINDOW_SIZE + 20:
        raise ValueError(f"Недостаточно данных для {ticker}")

    return returns


# =========================
# ПОДГОТОВКА ПОСЛЕДОВАТЕЛЬНОСТЕЙ
# =========================
def create_sequences(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for i in range(window_size, len(values)):
        X.append(values[i - window_size:i])
        y.append(values[i])

    return np.array(X), np.array(y)


# =========================
# МОДЕЛЬ
# =========================
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


# =========================
# ГРАФИК
# =========================
def save_prediction_plot(
    ticker: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Реальная доходность")
    plt.plot(y_pred, label="Прогноз LSTM")
    plt.title(f"LSTM прогноз доходности для {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================
# ОБУЧЕНИЕ ОДНОГО ТИКЕРА
# =========================
def train_one_ticker(ticker: str) -> Dict:
    print(f"\n{'=' * 70}")
    print(f"Обработка тикера: {ticker}")
    print(f"{'=' * 70}")

    split_idx_raw = int(len(returns) * (1 - TEST_SPLIT))

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

    if len(X_train) < 50 or len(X_test) < 10:
        raise ValueError(f"Слишком мало последовательностей для {ticker}")

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = build_model(WINDOW_SIZE)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    pred_scaled = model.predict(X_test, verbose=0)

    y_pred = scaler.inverse_transform(pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    last_window = scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    next_return_scaled = model.predict(last_window, verbose=0)
    next_return = scaler.inverse_transform(next_return_scaled)[0][0]

    model_path = os.path.join(MODEL_DIR, f"{ticker}_lstm.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{ticker}_scaler.pkl")
    plot_path = os.path.join(PLOTS_DIR, f"{ticker}_plot.png")

    model.save(model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    save_prediction_plot(ticker, y_true, y_pred, plot_path)

    best_val_loss = float(min(history.history["val_loss"]))
    final_train_loss = float(history.history["loss"][-1])
    epochs_trained = len(history.history["loss"])

    print(f"Готово: {ticker}")
    print(f"MSE: {mse:.8f}")
    print(f"MAE: {mae:.8f}")
    print(f"Прогноз следующей доходности: {next_return:.6f}")
    print(f"Модель сохранена: {model_path}")
    print(f"Scaler сохранён: {scaler_path}")
    print(f"График сохранён: {plot_path}")

    return {
        "ticker": ticker,
        "status": "success",
        "rows": len(returns),
        "samples": len(X_train) + len(X_test),
        "epochs_trained": epochs_trained,
        "train_loss_last": final_train_loss,
        "val_loss_best": best_val_loss,
        "mse": float(mse),
        "mae": float(mae),
        "next_return_forecast": float(next_return),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "plot_path": plot_path,
        "error": ""
    }


# =========================
# MAIN
# =========================
def main():
    results: List[Dict] = []

    for ticker in TICKERS:
        try:
            result = train_one_ticker(ticker)
        except Exception as e:
            print(f"Ошибка для {ticker}: {e}")
            result = {
                "ticker": ticker,
                "status": "failed",
                "rows": 0,
                "samples": 0,
                "epochs_trained": 0,
                "train_loss_last": None,
                "val_loss_best": None,
                "mse": None,
                "mae": None,
                "next_return_forecast": None,
                "model_path": "",
                "scaler_path": "",
                "plot_path": "",
                "error": str(e)
            }

        results.append(result)

        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig")

    print(f"\n{'#' * 70}")
    print("Обучение завершено")
    print(f"Результаты сохранены в: {RESULTS_FILE}")
    print(f"{'#' * 70}")

    df = pd.DataFrame(results)
    success_count = (df["status"] == "success").sum()
    failed_count = (df["status"] == "failed").sum()

    print(f"Успешно обучено: {success_count}")
    print(f"С ошибками: {failed_count}")

    if success_count > 0:
        ok_df = df[df["status"] == "success"].copy()
        print("\nТоп-10 по минимальному MAE:")
        print(
            ok_df.sort_values("mae")
            [["ticker", "mae", "mse", "next_return_forecast", "epochs_trained"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()