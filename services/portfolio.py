from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns, objective_functions
from pypfopt.discrete_allocation import DiscreteAllocation

from config import MIN_WEIGHT, MAX_WEIGHT
from services.lstm_forecaster import predict_next_return


TRADING_DAYS = 252

# Ограничение "разумности" ожидаемой годовой доходности
MU_CLIP_LOWER = -0.25
MU_CLIP_UPPER = 0.45


def _annualize_daily_return(daily_return: float) -> float:
    """
    Перевод дневной доходности в годовую.
    """
    if daily_return <= -0.999:
        return MU_CLIP_LOWER

    annualized = (1.0 + daily_return) ** TRADING_DAYS - 1.0
    return float(annualized)


def _compute_market_stress(close_data: pd.DataFrame) -> float:
    """
    Оценка стрессового режима рынка на основе:
    - всплеска краткосрочной волатильности относительно более длинной
    - текущей просадки по равновзвешенной корзине активов

    Возвращает число от 0 до 1.
    """
    returns = close_data.pct_change().dropna()
    if returns.empty or len(returns) < 60:
        return 0.0

    # Равновзвешенная "рыночная" серия по выбранным активам
    basket_returns = returns.mean(axis=1)

    short_window = min(21, len(basket_returns))
    long_window = min(126, len(basket_returns))

    short_vol = basket_returns.tail(short_window).std() * np.sqrt(TRADING_DAYS)
    long_vol = basket_returns.tail(long_window).std() * np.sqrt(TRADING_DAYS)

    if pd.isna(short_vol) or pd.isna(long_vol) or long_vol <= 1e-12:
        vol_stress = 0.0
    else:
        vol_ratio = short_vol / long_vol
        # ratio 1.0 -> 0 стресс, 2.5+ -> 1 стресс
        vol_stress = np.clip((vol_ratio - 1.0) / 1.5, 0.0, 1.0)

    # Просадка по корзине
    price_index = (1.0 + basket_returns).cumprod()
    running_max = price_index.cummax()
    drawdown = (price_index / running_max - 1.0).iloc[-1]

    # drawdown 0 -> 0, drawdown -20% -> 1
    dd_stress = np.clip(abs(min(drawdown, 0.0)) / 0.20, 0.0, 1.0)

    stress = 0.6 * vol_stress + 0.4 * dd_stress
    return float(np.clip(stress, 0.0, 1.0))


def _prepare_expected_returns(close_data: pd.DataFrame) -> pd.Series:
    """
    Гибридная оценка ожидаемой доходности:
    1. Долгосрочная историческая EMA-доходность
    2. Более свежая краткосрочная EMA-доходность
    3. Прогноз LSTM

    При высоком стресс-режиме:
    - уменьшается вклад LSTM
    - уменьшается вклад краткосрочной оценки
    - итоговая доходность дополнительно сжимается к более консервативной
    """
    # 1) Долгосрочная оценка
    long_mu = expected_returns.ema_historical_return(close_data, span=180)
    long_mu = long_mu.clip(lower=MU_CLIP_LOWER, upper=MU_CLIP_UPPER)

    # 2) Краткосрочная оценка по последнему окну
    recent_slice = close_data.tail(min(len(close_data), 126))
    recent_mu = expected_returns.ema_historical_return(recent_slice, span=45)
    recent_mu = recent_mu.clip(lower=MU_CLIP_LOWER, upper=MU_CLIP_UPPER)

    # 3) Прогнозы LSTM
    lstm_mu_dict = {}
    for ticker in close_data.columns:
        try:
            daily_pred = predict_next_return(ticker)
            annualized_pred = _annualize_daily_return(daily_pred)
            lstm_mu_dict[ticker] = annualized_pred
        except Exception:
            # Если нет модели или ошибка прогноза — fallback на long_mu
            lstm_mu_dict[ticker] = float(long_mu[ticker])

    lstm_mu = pd.Series(lstm_mu_dict, index=close_data.columns)
    lstm_mu = lstm_mu.clip(lower=MU_CLIP_LOWER, upper=MU_CLIP_UPPER)

    # 4) Оценка стрессового режима
    stress = _compute_market_stress(close_data)

    # 5) Динамические веса компонентов
    # В спокойном режиме:
    # long 0.50, recent 0.25, lstm 0.25
    # В стрессе больше доверяем долгой истории и меньше LSTM
    w_long = 0.50 + 0.20 * stress
    w_recent = 0.25 - 0.10 * stress
    w_lstm = 0.25 - 0.10 * stress

    total = w_long + w_recent + w_lstm
    w_long /= total
    w_recent /= total
    w_lstm /= total

    mu = w_long * long_mu + w_recent * recent_mu + w_lstm * lstm_mu

    # 6) Дополнительное сжатие доходностей при стрессовом рынке
    shrink_factor = 1.0 - 0.35 * stress
    mu = mu * shrink_factor

    # 7) Финальный clip
    mu = mu.clip(lower=MU_CLIP_LOWER, upper=MU_CLIP_UPPER)

    return mu


def _prepare_covariance(close_data: pd.DataFrame) -> pd.DataFrame:
    """
    Более устойчивая ковариационная матрица.
    """
    cs = risk_models.CovarianceShrinkage(close_data)
    return cs.ledoit_wolf()


def _calculate_portfolio_metrics(
    weights: Dict[str, float],
    mu: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """
    Считает доходность, риск и Sharpe для заданных весов.
    """
    tickers = [ticker for ticker, weight in weights.items() if weight > 0]
    if not tickers:
        return 0.0, 0.0, 0.0

    w = np.array([weights[t] for t in tickers], dtype=float)
    mu_vec = mu.loc[tickers].values
    cov = cov_matrix.loc[tickers, tickers].values

    portfolio_return = float(np.dot(w, mu_vec))
    portfolio_volatility = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

    if portfolio_volatility <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((portfolio_return - risk_free_rate) / portfolio_volatility)

    return portfolio_return, portfolio_volatility, sharpe


def optimize_portfolios(close_data: pd.DataFrame, risk_free_rate: float) -> Dict[str, Any]:
    if close_data.shape[1] < 2:
        raise ValueError("Для оптимизации нужно минимум 2 актива.")

    mu = _prepare_expected_returns(close_data)
    s = _prepare_covariance(close_data)

    # max sharpe
    ef_sharpe = EfficientFrontier(mu, s, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
    ef_sharpe.add_objective(objective_functions.L2_reg, gamma=0.15)
    ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    sharpe_weights = ef_sharpe.clean_weights()
    sharpe_theoretical_performance = ef_sharpe.portfolio_performance(
        verbose=False,
        risk_free_rate=risk_free_rate
    )

    # min volatility
    ef_min_vol = EfficientFrontier(mu, s, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
    ef_min_vol.add_objective(objective_functions.L2_reg, gamma=0.15)
    ef_min_vol.min_volatility()
    min_vol_weights = ef_min_vol.clean_weights()
    min_vol_theoretical_performance = ef_min_vol.portfolio_performance(
        verbose=False,
        risk_free_rate=risk_free_rate
    )

    return {
        "expected_returns": mu,
        "cov_matrix": s,
        "market_stress": _compute_market_stress(close_data),
        "max_sharpe": {
            "weights": sharpe_weights,
            "theoretical_performance": sharpe_theoretical_performance,
        },
        "min_volatility": {
            "weights": min_vol_weights,
            "theoretical_performance": min_vol_theoretical_performance,
        },
    }


def discrete_allocate(
    weights: Dict[str, float],
    latest_prices: pd.Series,
    total_budget: float,
) -> Dict[str, Any]:
    non_zero_weights = {ticker: weight for ticker, weight in weights.items() if weight > 0}

    if not non_zero_weights:
        raise ValueError("Оптимизация вернула нулевые веса для всех активов.")

    da = DiscreteAllocation(
        non_zero_weights,
        latest_prices,
        total_portfolio_value=total_budget,
    )
    allocation, leftover = da.greedy_portfolio()

    spent = 0.0
    positions = []

    for ticker, shares in allocation.items():
        price = float(latest_prices[ticker])
        cost = price * shares
        spent += cost
        positions.append({
            "ticker": ticker,
            "shares": int(shares),
            "price": price,
            "cost": cost,
            "target_weight": float(non_zero_weights[ticker]),
        })

    positions.sort(key=lambda x: x["cost"], reverse=True)

    # фактические веса уже после округления
    actual_weights = {}
    if total_budget > 0:
        for pos in positions:
            actual_weights[pos["ticker"]] = pos["cost"] / total_budget

    return {
        "positions": positions,
        "leftover": float(leftover),
        "spent": float(spent),
        "budget": float(total_budget),
        "actual_weights": actual_weights,
    }


def build_full_portfolio_report(close_data: pd.DataFrame, budget: float, risk_free_rate: float) -> Dict[str, Any]:
    latest_prices = close_data.iloc[-1]
    optimized = optimize_portfolios(close_data, risk_free_rate=risk_free_rate)

    mu = optimized["expected_returns"]
    cov_matrix = optimized["cov_matrix"]

    max_sharpe_alloc = discrete_allocate(
        optimized["max_sharpe"]["weights"],
        latest_prices,
        budget,
    )
    min_vol_alloc = discrete_allocate(
        optimized["min_volatility"]["weights"],
        latest_prices,
        budget,
    )

    max_sharpe_actual_perf = _calculate_portfolio_metrics(
        max_sharpe_alloc["actual_weights"],
        mu,
        cov_matrix,
        risk_free_rate,
    )
    min_vol_actual_perf = _calculate_portfolio_metrics(
        min_vol_alloc["actual_weights"],
        mu,
        cov_matrix,
        risk_free_rate,
    )

    return {
        "latest_prices": latest_prices.to_dict(),
        "market_stress": optimized["market_stress"],
        "max_sharpe": {
            "weights": optimized["max_sharpe"]["weights"],
            "theoretical_performance": optimized["max_sharpe"]["theoretical_performance"],
            "actual_performance": max_sharpe_actual_perf,
            "allocation": max_sharpe_alloc,
        },
        "min_volatility": {
            "weights": optimized["min_volatility"]["weights"],
            "theoretical_performance": optimized["min_volatility"]["theoretical_performance"],
            "actual_performance": min_vol_actual_perf,
            "allocation": min_vol_alloc,
        },
        "mu_used": mu.to_dict(),
    }