from __future__ import annotations

from typing import Dict, Any


def format_weights(weights: Dict[str, float]) -> str:
    lines = []
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        if weight > 0:
            lines.append(f"• {ticker}: {weight * 100:.2f}%")
    return "\n".join(lines) if lines else "Нет ненулевых весов."


def format_actual_weights(weights: Dict[str, float]) -> str:
    if not weights:
        return "Нет фактических позиций."
    lines = []
    for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"• {ticker}: {weight * 100:.2f}%")
    return "\n".join(lines)


def format_portfolio_block(title: str, portfolio_data: Dict[str, Any]) -> str:
    th_return, th_volatility, th_sharpe = portfolio_data["theoretical_performance"]
    act_return, act_volatility, act_sharpe = portfolio_data["actual_performance"]
    allocation = portfolio_data["allocation"]

    lines = [
        f"{title}",
        "",
        "Теоретическая оценка по непрерывным весам:",
        f"• Ожидаемая годовая доходность: {th_return * 100:.2f}%",
        f"• Годовой риск (волатильность): {th_volatility * 100:.2f}%",
        f"• Коэффициент Шарпа: {th_sharpe:.3f}",
        "",
        "Теоретические веса:",
        format_weights(portfolio_data["weights"]),
        "",
        "Реальная покупка по штукам:",
    ]

    if allocation["positions"]:
        for pos in allocation["positions"]:
            lines.append(
                f"• {pos['ticker']}: {pos['shares']} шт. × ${pos['price']:.2f} = ${pos['cost']:.2f}"
            )
    else:
        lines.append("• Не удалось собрать дискретный портфель.")

    lines.extend([
        "",
        "Фактическая структура после округления:",
        format_actual_weights(allocation["actual_weights"]),
        "",
        "Оценка уже для реально покупаемого портфеля:",
        f"• Ожидаемая годовая доходность: {act_return * 100:.2f}%",
        f"• Годовой риск (волатильность): {act_volatility * 100:.2f}%",
        f"• Коэффициент Шарпа: {act_sharpe:.3f}",
        "",
        f"Инвестировано: ${allocation['spent']:.2f}",
        f"Остаток денежных средств: ${allocation['leftover']:.2f}",
        "",
        "Примечание: это оценка на основе исторических данных и модели оптимизации, а не гарантия будущей доходности.",
    ])

    return "\n".join(lines)


def format_full_portfolio_report(report: Dict[str, Any]) -> str:
    max_sharpe_text = format_portfolio_block(
        "📈 Портфель с максимальным коэффициентом Шарпа",
        report["max_sharpe"],
    )
    min_vol_text = format_portfolio_block(
        "🛡 Портфель с минимальной волатильностью",
        report["min_volatility"],
    )

    return f"{max_sharpe_text}\n\n{'—' * 35}\n\n{min_vol_text}"