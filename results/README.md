# Результаты экспериментов

Текущая проверка использует 12 задач UserBench. Объяснение результатов находится в [разборе E027](../research/E027_ERROR_ANALYSIS.md).

| Файл или каталог | Содержание |
|---|---|
| [userbench_executor_e027/summary.json](userbench_executor_e027/summary.json) | Сводные метрики Qwen, эвристики и случайного выбора |
| [userbench_executor_e027/generations.jsonl](userbench_executor_e027/generations.jsonl) | Все 192 ответа Qwen и параметры выполнения |
| [userbench_executor_e027/question_outcomes.json](userbench_executor_e027/question_outcomes.json) | Изменение качества после каждого вопроса |
| [userbench_executor_e027/ranking_details.json](userbench_executor_e027/ranking_details.json) | Лучшие вопросы, ничьи и сравнение полных и частичных ответов |
| [e027_audit.json](e027_audit.json) | Отдельный пересчёт и нарушения предпочтений |
| [userbench_train_e025](userbench_train_e025) | Подготовленные задачи и исходные метки |
| [userbench_semantics_e026](userbench_semantics_e026) | Предварительные правила проверки предпочтений |

JSON хранит точные значения, в том числе дроби. Графики и пояснения доступны в [EDA](../EDA.ipynb). Команды проверки находятся в [REPRODUCE.md](../REPRODUCE.md).

Остальные каталоги относятся к предыдущим экспериментам. Каждый набор следует читать вместе с соответствующим протоколом в research/.
