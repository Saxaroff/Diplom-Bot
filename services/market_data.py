from __future__ import annotations

from typing import List
import pandas as pd
import yfinance as yf

from config import DEFAULT_RISK_FREE_RATE


def normalize_tickers(raw_tickers: List[str]) -> List[str]:
    cleaned = []
    for ticker in raw_tickers:
        ticker = ticker.strip().upper()
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    return cleaned


def download_close_prices(tickers: List[str], period: str) -> pd.DataFrame:
    if not tickers:
        raise ValueError("Список тикеров пуст.")

    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    if data.empty:
        raise ValueError("Не удалось загрузить данные по тикерам.")

    if len(tickers) == 1:
        ticker = tickers[0]
        if "Close" not in data.columns:
            raise ValueError(f"Не найден столбец Close для {ticker}.")
        close_data = data[["Close"]].copy()
        close_data.columns = [ticker]
    else:
        close_frames = {}
        for ticker in tickers:
            if ticker in data.columns.get_level_values(0):
                ticker_block = data[ticker]
                if "Close" in ticker_block.columns:
                    close_frames[ticker] = ticker_block["Close"]
        if not close_frames:
            raise ValueError("Не удалось извлечь цены закрытия.")
        close_data = pd.DataFrame(close_frames)

    close_data = close_data.dropna(axis=1, how="all")
    close_data = close_data.ffill().bfill().dropna(axis=0, how="any")

    if close_data.empty:
        raise ValueError("После очистки не осталось данных для расчёта.")

    if close_data.shape[1] < 2:
        raise ValueError("Для портфельного анализа нужно минимум 2 тикера с корректными данными.")

    return close_data


def get_latest_prices_series(close_data: pd.DataFrame) -> pd.Series:
    if close_data.empty:
        raise ValueError("Нет данных цен для определения последних котировок.")
    return close_data.iloc[-1]


def get_risk_free_rate() -> float:
    try:
        rf_data = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        close = rf_data["Close"].dropna()
        if close.empty:
            return DEFAULT_RISK_FREE_RATE
        return float(close.iloc[-1]) / 100.0
    except Exception:
        return DEFAULT_RISK_FREE_RATE


def get_stock_info(ticker: str):
    return yf.Ticker(ticker)


def get_stock_history(ticker: str, period: str) -> pd.Series:
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history.columns:
        raise ValueError(f"Не удалось получить историю для {ticker}.")
    return history["Close"]