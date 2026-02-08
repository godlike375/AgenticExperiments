"""
### Критические требования (ядро проекта)

1. Цель — поиск максимально универсального короткого системного промпта**  (желательно до 200 символов) системный промпт,
 который превращает обычную instruct-LLM в сильную reasoning-LLM на **широком спектре задач.

2. Первичный компонент фитнеса — **доля правильно решённых задач** (accuracy).
Все остальные метрики (длина промпта, длина ответов) — вторичные и могут только штрафовать, но не перевешивать accuracy.

3. Полное отсутствие few-shot примеров в системных промптах выдаваемых Generatorом для Solverа
Нигде **не должно быть** few-shot примеров (примеры задач + решений). Это дорого по токенам и снижает универсальность.

4. Generator не должен видеть сами задачи и правильные ответы
- Generator получает только статистику (accuracy, critic-комментарии, лидеров).
Это обязательное условие, чтобы избежать подгонки и подсказок под конкретный тестовый набор.

5. Judge должен возвращать строго валидный JSON
Судья обязан выводить только JSON-объект с полями:
- `"correct": "TRUE" | "FALSE"`
- `"confidence": float 0.0–1.0`
Любое отклонение от формата → fallback на INCORRECT.

6. Harmony / gpt-oss поддержка должна быть полной
- Функция `_clean_harmony_response` обязательно должна удалять:
- служебные токены `<|start|>`, `<|end|>` и т.д.
- парные теги `<think>…</think>`, `<reasoning>…</reasoning>` и аналоги
- любые оставшиеся одиночные теги
- Важно: очистка не должна удалять обычные слова "think" и т.п. из текста.

7. Ответ Solverа должен очищаться от служебных тегов перед передачей судье

8. Надёжный и устойчивый парсинг новых промптов
Функция `extract_prompts` должна корректно извлекать промпты при любом из выбранных форматов (`json` или `xml`) и быть устойчивой к небольшим ошибкам модели (лишние пробелы, переносы строк, отсутствие кавычек и т.д.).

9. Элитизм обязателен
Лучший промпт поколения всегда сохраняется в следующее поколение (elite_size ≥ 1).
Нельзя допускать регресса по accuracy больше определенного значения.

10. Температуры должны быть адекватными ролям
- Judge → 0.0 (максимальная детерминированность)
- Solver → низкая (0.3)
- Generator → высокая (0.9) для разнообразия

11. Остановка по высокому уровню универсальности
Условие остановки должно учитывать accuracy == 1.0
"""


import json
import re
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

base_url = "http://localhost:1234/v1"
client = OpenAI(base_url=base_url, api_key="lm-studio")


# ────────────────────────────────────────────────
# Агент с поддержкой Harmony / gpt-oss
# ────────────────────────────────────────────────

class Agent:
    def __init__(
            self,
            name: str,
            system_prompt: str = "",
            default_temperature: float = 0.7,
            memory: Optional[List[Dict]] = None,
            harmony_mode: bool = True,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.default_temperature = default_temperature
        self.memory = memory or []
        self.harmony_mode = harmony_mode

    def _clean_harmony_response(self, text: str) -> str:
        if not self.harmony_mode:
            return text

        text = re.sub(r'<\|start\|>|<\|end\|>|<\|message\|>|<\|im_end\|>|<\|eot_id\|>', '', text)
        text = re.sub(r'<(think|analysis|reasoning|internal)[^>]*>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)

        return text.strip()

    def chat(
            self,
            user_content: str,
            temperature: Optional[float] = None,
            max_tokens: int = 2048,
            label_suffix: str = "",
    ) -> str:
        temperature = temperature if temperature is not None else self.default_temperature

        messages = [{"role": "system", "content": self.system_prompt}] + self.memory + [
            {"role": "user", "content": user_content}
        ]

        print(f"\n--- {self.name}{label_suffix} ---")
        full_response = ""
        stream = client.chat.completions.create(
            model="local-model",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
        print("\n")

        cleaned = self._clean_harmony_response(full_response)

        if self.harmony_mode and cleaned != full_response.strip():
            print(f"[Harmony cleaned {len(full_response) - len(cleaned)} chars]")

        return cleaned


# ────────────────────────────────────────────────
# Глобальная настройка формата вывода промптов
# ────────────────────────────────────────────────

PROMPT_OUTPUT_FORMAT = "json"  # "json" или "xml" — легко переключить здесь или через функцию

def set_prompt_output_format(fmt: str = "json"):
    """
    Устанавливает формат вывода промптов для Generator и Mutator.
    """
    global PROMPT_OUTPUT_FORMAT
    PROMPT_OUTPUT_FORMAT = fmt.lower()

    common_generator = (
        "Ты создаёшь короткие (60–140 слов) универсальные CoT-промпты для instruct-LLM.\n"
    )

    if fmt == "json":
        generator_agent.system_prompt = (
                common_generator +
                "ВЫВОДИ СТРОГО ТОЛЬКО ОДИН ВАЛИДНЫЙ JSON-ОБЪЕКТ БЕЗ ЛЮБОГО ДРУГОГО ТЕКСТА:\n"
                '{"prompt": "текст промпта"}\n'
        )

    elif fmt == "xml":
        generator_agent.system_prompt = (
                common_generator +
                "ВЫВОДИ ТОЛЬКО ТЕГИ <prompt>текст промпта</prompt> БЕЗ ЛЮБОГО ДРУГОГО ТЕКСТА\n"
        )

    else:
        raise ValueError("Формат должен быть 'json' или 'xml'")


# ────────────────────────────────────────────────
# Надёжный парсер промптов в зависимости от формата
# ────────────────────────────────────────────────

def extract_prompts(text: str, fmt: str) -> List[str]:
    text = text.strip()
    if fmt == "json":
        try:
            data = json.loads(text)
            prompts = data.get("prompts", [])
            return [p.strip() for p in prompts if p.strip()]
        except json.JSONDecodeError:
            print("[Ошибка парсинга JSON от модели]")
            return []
    elif fmt == "xml":
        matches = re.findall(r"<prompt>(.*?)</prompt>", text, re.DOTALL | re.IGNORECASE)
        return [m.strip() for m in matches if m.strip()]
    return []


# ────────────────────────────────────────────────
# Тестовый набор задач (ОБЯЗАТЕЛЬНО расширьте до 12–20+)
# ────────────────────────────────────────────────

TASKS = [
    {
        "id": 1,
        "text": (
            "Билет на поезд с надписью: \"Москва → Владивосток, отправление 15.08.1975, 08:00\".\n"
            "На обратной стороне — заметка:\n"
            "\"Прибыл на 6 часов позже из-за аварии на БАМе. Встретил сына, которому в тот день исполнилось 18 лет. "
            "Сегодня 27 января 2026 года, и мне сейчас 93 года. Я родился в столице советской республики, "
            "которая в 1991 году стала независимым государством и граничит с Китаем и Казахстаном. "
            "Я был на 25 лет старше сына в день его 18-летия.\"\n\n"
            "В каком городе (укажите название, действовавшее на момент рождения) и какого числа родился владелец сейфа?\n"
            "Ответ форматируйте как: \"Город, ДД.ММ.ГГГГ\""
        ),
        "answer": "Фрунзе, 15.08.1932"
    },
    # {
    #     "id": 2,
    #     "text": (
    #         "Запись в дневнике от 12 июля 1989 года:\n"
    #         "\"Сегодня в 15:00 по местному времени я вылетел из Буэнос-Айреса в Токио. Рейс занял ровно 24 часа. "
    #         "По прибытии в Токио я сразу позвонил сыну, который находился в Лондоне.\"\n\n"
    #         "Какое время и дата были в Лондоне в момент моего звонка?\n"
    #         "Ответ форматируйте как: \"ДД.ММ.ГГГГ ЧЧ:ММ\""
    #     ),
    #     "answer": "13.07.1989 19:00"
    # },
    # {
    #     "id": 3,
    #     "text": (
    #         "Я купил 100 акций Apple 12 декабря 1980 года по цене $22 за акцию. "
    #         "На 31 декабря 2025 года цена одной акции составляла $185. "
    #         "Если бы я продал все акции на эту дату, какова была бы моя чистая прибыль в долларах США, "
    #         "учитывая все сплиты акций Apple до 2025 года?\n"
    #         "Ответ округлите до ближайшего целого числа и дайте только цифру."
    #     ),
    #     "answer": "4141800"
    # },
    # {
    #     "id": 4,
    #     "text": (
    #         "Пациенту 58 лет. Обнаружена генетическая мутация BRCA1 — передается по наследству с вероятностью 50%. "
    #         "У пациента есть сын, которому в 1991 году исполнилось 22 года. "
    #         "Сын женится 15 июня 1991 года на женщине без мутации BRCA1. У них планируется двое детей. "
    #         "Какова вероятность (в процентах), что хотя бы один из будущих внуков унаследует мутацию BRCA1?\n"
    #         "Ответ дайте только цифрой без знака % (округлите до ближайшего целого)."
    #     ),
    #     "answer": "38"
    # },
    # {
    #     "id": 5,
    #     "text": (
    #         "Наследство делится согласно швейцарскому праву, но с учетом принципов немецкого наследственного права.\n"
    #         "Код доступа рассчитывается так:\n"
    #         "1. Возьмите год принятия Гражданского кодекса Германии (BGB).\n"
    #         "2. Прибавьте количество лет, прошедших с основания Швейцарской Конфедерации (1291) до вступления Швейцарии в ООН (2002).\n"
    #         "3. Умножьте результат на количество официальных языков Европейского Союза в 2026 году.\n"
    #         "4. Разделите на количество государств-членов ООН в 2026 году (округлите вниз).\n"
    #         "5. Прибавьте разницу между годом вступления Германии в ООН (1973) и годом вступления Швейцарии в ООН (2002).\n"
    #         "6. Если полученная сумма делится на 3 без остатка — прибавьте количество кантонов Швейцарии (26). Иначе — вычтите количество земель Германии (16).\n"
    #         "7. Если результат меньше 1000 — умножьте его на 3 и прибавьте 1945 (год основания ООН).\n\n"
    #         "Полученный 4-значный код — это код доступа к ячейке.\n"
    #         "Какой 4-значный код нужно ввести?"
    #     ),
    #     "answer": "2956"
    # }
]


# ────────────────────────────────────────────────
# Оценка и судья
# ────────────────────────────────────────────────

def evaluate_prompt(solver: Agent, prompt: str) -> Dict[str, Any]:
    results = []
    correct_count = 0
    total_response_len = 0

    for task in TASKS:
        print(f"\nEval task {task['id']} ...")
        answer = solver.chat(task["text"], label_suffix=f" — task {task['id']}")
        total_response_len += len(answer)

        judgment_raw = judge_agent.chat(
            f"Сравни ответ агента:\n{answer}\n\nс правильным эталоном:\n{task['answer']}\n",
            label_suffix=f" — task {task['id']}"
        )
        judgment = parse_judge_output(judgment_raw)

        is_correct = judgment.get("correct", "").upper() == "TRUE"
        if is_correct:
            correct_count += 1

        results.append({
            "task_id": task["id"],
            "correct": is_correct,
        })

    accuracy = correct_count / len(TASKS) if TASKS else 0
    avg_response_len = total_response_len / len(TASKS) if TASKS else 0

    fitness = (
        accuracy,
        -len(prompt) / 1000.0,
        -avg_response_len / 2000.0
    )

    return {
        "fitness": fitness,
        "accuracy": accuracy,
        "prompt_len": len(prompt),
        "avg_response_len": avg_response_len,
        "correct_count": correct_count,
        "total": len(TASKS),
        "details": results
    }


def parse_judge_output(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except:
        return {"correct": "FALSE", "confidence": 0.0}


# ────────────────────────────────────────────────
# Агенты
# ────────────────────────────────────────────────

judge_agent = Agent(
    name="Judge",
    system_prompt=(
        "Ты — строгий судья. Выведи ТОЛЬКО валидный JSON:\n"
        '{"correct": "TRUE" или "FALSE", "confidence": 0.0–1.0}'
    ),
    default_temperature=0.0,
)

critic_agent = Agent(
    name="Critic",
    system_prompt=(
        "Кратко анализируй статистику промпта. Опиши типичные слабые места и дай 2-3 рекомендации по улучшению"
    ),
    default_temperature=0.75,
)

generator_agent = Agent(
    name="Generator",
    system_prompt="",  # будет установлен в set_prompt_output_format
    default_temperature=0.9,
)

solver_agent = Agent(
    name="Solver",
    system_prompt="",
    default_temperature=0.25,
)


# ────────────────────────────────────────────────
# Основной цикл эволюции
# ────────────────────────────────────────────────

def run_evolution(
        generations: int = 20,
        population_size: int = 6,
        elite_size: int = 1,
        top_for_breed: int = 2,
        mutation_rate: float = 0.35,
) -> Tuple[str, List[Dict]]:
    # Устанавливаем формат вывода промптов (можно изменить здесь или вызвать set_prompt_output_format("xml"))
    set_prompt_output_format(PROMPT_OUTPUT_FORMAT)

    history: List[Dict] = []

    current_population = [
                             "Думай шаг за шагом. Проверяй логику на каждом этапе.",
                             #"Сначала факты → предположения → цепочка рассуждений → ответ.",
                             #"Рассуждай вслух. Избегай поспешных заключений.",
                             #"Планируй решение. Пиши финальный ответ только после проверки.",
                         ][:population_size]

    best_fitness = (-1.0, 0.0, 0.0)
    best_prompt = current_population[0]

    main_system_solver_prompt = solver_agent.system_prompt

    for gen in range(1, generations + 1):
        print(f"\n{'═' * 80}\nПОКОЛЕНИЕ {gen}\n{'═' * 80}")

        gen_results = []

        for i, prompt in enumerate(current_population, 1):
            solver_agent.system_prompt = main_system_solver_prompt + f"\n{prompt}"
            eval_result = evaluate_prompt(solver_agent, prompt)
            critic_text = critic_agent.chat(
                f"Промпт:\n{prompt}\n\nРезультаты:\n{json.dumps(eval_result, indent=2, ensure_ascii=False)}"
            )

            entry = {
                "generation": gen,
                "prompt": prompt,
                "fitness": eval_result["fitness"],
                "accuracy": eval_result["accuracy"],
                "critic": critic_text,
            }
            history.append(entry)
            gen_results.append((prompt, eval_result["fitness"], eval_result))

            fit = eval_result["fitness"]
            print(f"  Промпт #{i} | acc={fit[0]:.3f} | len={eval_result['prompt_len']} | resp≈{eval_result['avg_response_len']:.0f}")
            if fit > best_fitness:
                best_fitness = fit
                best_prompt = prompt
                print("  ← новый рекорд!")

        gen_results.sort(key=lambda x: x[1], reverse=True)

        elites = [p for p, _, _ in gen_results[:elite_size]]
        breeders = [p for p, _, _ in gen_results[:top_for_breed]]

        print(f"  Лучший: acc={best_fitness[0]:.3f}, len={len(best_prompt)}")

        if best_fitness[0] >= 0.92 and len(TASKS) >= 10:
            print("\nДостигнут высокий уровень!")
            break

        n_new = population_size - len(elites)
        n_mutate = max(1, int(population_size * mutation_rate))

        history_summary = json.dumps(
            [{"gen": e["generation"], "acc": e["accuracy"], "critic": e["critic"][:140]+"..."} for e in history[-10:]],
            ensure_ascii=False
        )

        # Один вызов → несколько промптов в структурированном формате
        new_prompts = []
        for _ in range(n_new):
            new_prompt_raw = generator_agent.chat(
                f"История последних:\n{history_summary}\n\n"
                f"Лидеры:\n{chr(10).join([f'- {p[:100]}...' for p in breeders])}\n\n"
                f"Создай новый CoT-промпт ориентируясь на эти данные. Не пиши ничего лишнего помимо промпта"
            )
            new_prompts.append(new_prompt_raw)

        # Мутации
        if breeders:
            new_mutants = []
            for _ in range(n_mutate):
                mutate_from = breeders[0]
                mutate_raw = generator_agent.chat(
                    f"Исходный промпт:\n{mutate_from}\n\n"
                    f"Создай его вариацию в ответе. Не пиши ничего лишнего помимо промпта"
                )
                new_mutants.append(mutate_raw)
            for p in new_mutants:
                if p not in new_prompts and p not in elites:
                    new_prompts.append(p)

        # Fallback
        fallback = "Думай тщательно шаг за шагом. Проверяй каждый шаг."
        while len(new_prompts) < n_new:
            new_prompts.append(fallback)

        current_population = elites + new_prompts[:n_new]

    print(f"\nЛучший промпт (fitness {best_fitness}):")
    print(best_prompt)

    return best_prompt, history


if __name__ == "__main__":
    # Чтобы переключить на XML-формат — раскомментируйте:
    # set_prompt_output_format("xml")

    best, hist = run_evolution(
        generations=20,
        population_size=6,
        elite_size=1,
        top_for_breed=2,
        mutation_rate=0.35,
    )