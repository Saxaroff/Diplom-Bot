import json
import os
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.groq.com/openai/v1"
)


def explain_portfolio_with_llm(portfolio_payload: dict, news_context: str = "") -> str:
    max_sharpe = portfolio_payload.get("max_sharpe", {})
    min_vol = portfolio_payload.get("min_volatility", {})

    def short_block(block: dict) -> dict:
        theoretical = block.get("theoretical_performance", (0, 0, 0))
        actual = block.get("actual_performance", (0, 0, 0))
        weights = block.get("weights", {})
        allocation = block.get("allocation", {})
        positions = allocation.get("positions", [])

        top_weights = sorted(
            [(k, v) for k, v in weights.items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        top_positions = [
            {
                "ticker": p["ticker"],
                "shares": p["shares"],
                "cost": round(p["cost"], 2)
            }
            for p in positions[:5]
        ]

        return {
            "theoretical_return_pct": round(theoretical[0] * 100, 2) if theoretical else 0,
            "theoretical_risk_pct": round(theoretical[1] * 100, 2) if theoretical else 0,
            "theoretical_sharpe": round(theoretical[2], 3) if theoretical else 0,
            "actual_return_pct": round(actual[0] * 100, 2) if actual else 0,
            "actual_risk_pct": round(actual[1] * 100, 2) if actual else 0,
            "actual_sharpe": round(actual[2], 3) if actual else 0,
            "top_weights": top_weights,
            "top_positions": top_positions,
            "leftover_usd": round(allocation.get("leftover", 0), 2),
        }

    short_payload = {
        "market_stress": round(portfolio_payload.get("market_stress", 0), 3),
        "max_sharpe": short_block(max_sharpe),
        "min_volatility": short_block(min_vol),
    }

    prompt = f"""
Ты профессиональный финансовый аналитик инвестиционной платформы.

Ниже представлены данные по двум инвестиционным портфелям:
1. Портфель с максимальным коэффициентом Шарпа.
2. Портфель с минимальной волатильностью.

Данные портфелей:

{json.dumps(short_payload, ensure_ascii=False, indent=2)}

Сформируй подробный аналитический отчет на русском языке.

Требования к анализу:

1. Сравни оба портфеля между собой.
2. Проанализируй уровень риска каждого портфеля.
3. Оцени ожидаемую доходность.
4. Объясни значение коэффициента Шарпа.
5. Проанализируй диверсификацию активов.
6. Укажи доминирующие активы и возможные риски концентрации.
7. Объясни влияние рыночного стресса.
8. Отдельно прокомментируй разницу между теоретическим и реально покупаемым портфелем.
9. Объясни смысл остатка денежных средств.
10. Укажи сильные и слабые стороны каждого варианта портфеля.
11. Сделай общий вывод:
    - какой портфель более агрессивный;
    - какой более консервативный;
    - для какого типа инвестора каждый подходит.
12. После аналитического вывода добавь отдельный блок:
"Пояснение финансовых терминов".

В этом блоке простым языком кратко объясни термины, которые использовались в анализе:
- диверсификация;
- волатильность;
- коэффициент Шарпа;
- рыночный стресс;
- доходность;
- риск портфеля.

Объяснения должны быть понятны начинающему инвестору.

Правила ответа:
- Пиши профессионально.
- Не используй markdown.
- Основной аналитический текст пиши без списков.
- Ответ должен выглядеть как аналитическая заметка финансового специалиста.
- Используй финансовую терминологию.
- Не придумывай данные, которых нет.
- Не давай прямых инвестиционных рекомендаций.
- Обязательно укажи, что результаты не гарантируют будущую прибыль.
- Объем ответа: 12–20 предложений.

!!!Блок "Пояснение финансовых терминов" ОБЯЗАТЕЛЬНО оформляй в виде списка.!!!
!!!• Термин — краткое и понятное объяснение.!!!
!!!Каждый термин должен начинаться с новой строки.!!!

Не объединяй несколько терминов в один абзац.
После основного анализа добавь подзаголовок:
"Пояснение финансовых терминов".
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Ты опытный финансовый аналитик, специализирующийся на оценке инвестиционных портфелей, анализе риска, диверсификации и интерпретации результатов портфельной оптимизации. Твоя задача — формировать профессиональные аналитические комментарии по рассчитанным инвестиционным портфелям."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI-анализ временно недоступен. Причина: {e}"