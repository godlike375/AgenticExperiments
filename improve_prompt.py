import time
from openai import OpenAI

base_url = "http://192.168.50.196:1234/v1"
client = OpenAI(base_url=base_url, api_key="lm-studio")

def stream_request(system_prompt, user_prompt, label):
    """Вспомогательная функция для стриминга с меткой агента"""
    print(f"\n--- {label} ---")
    full_response = ""
    stream = client.chat.completions.create(
        model="local-model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
            full_response += content
    print("\n")
    return full_response

def run_experiment(puzzle, correct_answer, iterations=3):
    history = [] # Здесь храним результаты попыток для Агента 1
    current_best_prompt = "Реши задачу."

    for i in range(iterations):
        print(f"=== ИТЕРАЦИЯ {i+1} ===")

        # 1. Агент 1 анализирует историю и создает/улучшает промпт
        history_str = "\n".join([f"Попытка {j+1}: Промпт: [{h['prompt']}] | Результат: {h['status']}" for j, h in enumerate(history)])

        a1_system = (
            "Ты — Агент 1, эксперт по промпт-инжинирингу. Твоя задача: составить короткий универсальный промпт для любых задач "
            "для Агента 2"
            f"Задача: {puzzle}. "
            "Анализируй предыдущие ошибки и подбирай наиболее универсальные инструкции, которые помогут Агенту 2 прийти к верному логическому выводу."
        )

        a1_user = f"История предыдущих тестов:\n{history_str}\n\nВыдай только текст нового системного промпта."
        current_best_prompt = stream_request(a1_system, a1_user, "АГЕНТ 1 (Генерация промпта)")

        # 2. Агент 2 пытается решить задачу с новым промптом
        a2_response = stream_request(current_best_prompt, puzzle, "АГЕНТ 2 (Решение задачи)")

        # 3. Агент 1 проверяет решение и собирает "статистику" (анализ)
        eval_system = (
            "Ты — судья. Сравни ответ Агента 2 с правильным ответом. "
            f"Правильный ответ: {correct_answer}. "
            "Вынеси вердикт: Проанализируй вслух и самым последним словом вынеси вердики - ВЕРНО или НЕВЕРНО"
        )

        evaluation = stream_request(eval_system, f"Ответ Агента 2: {a2_response}", "АГЕНТ 1 (Анализ и Статистика)")

        # Сохраняем данные для следующего цикла
        status = "ВЕРНО" if " ВЕРНО " in evaluation.upper() else "НЕВЕРНО"
        history.append({
            "prompt": current_best_prompt,
            "status": status,
            "analysis": evaluation
        })

        if status == "ВЕРНО":
            print("Цель достигнута! Оптимальный промпт найден.")
            break

# Данные для задачи
puzzle_text = "Есть широкий металлический стакан, у которого было отрезано дно и запаяна горловина. Как в него налить и из него пить?"
answer_text = "Перевернуть стакан вверх ногами, тогда он будет функционировать как обычный стакан (сверху открытое отверстие в роли горловины, снизу запаянная горловина в роли дна)"

run_experiment(puzzle_text, answer_text)