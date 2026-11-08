# E015: доступный текстовый оцениватель ценности вопроса

27 октября 2026. До fitting E015, после полного анализа E014. **Это exploratory development cross-validation, не новый слепой тест и не проверка публикационной новизны.**

## Данные и доступ

Все24 case-группы/65 скрытых профилей E014. На одну case-группу приходится один исходный видимый текст; проверено до fitting. Одинаковые тексты разных case-групп, если найдутся, объединяются в одну внешнюю fold-группу. Все профили одной case-группы остаются вместе. Источник labels — сохранённые390 branch outcomes фиксированного Qwen3-4B. Новых генераций нет.

Для каждой case получить6 targets: средняя точность по её профилям для no-question, issue, product, money_gone, compensation, customer_name. Это noisy observed accuracy, не истинный expected value. Полное ветвление используется для обучающих labels: его бюджет явно больше одного feedback на ситуацию. Источник истинных классов и исходов доступен evaluator, но не selector на тестовой группе.

Input модели — только первоначальный пользовательский текст. Список шести действий одинаков и известен. Запрещённые признаки: source ID, gold/profile, reply/answer_kind, evidence/reveal, posterior/oracle, текущие исходы исполнителя. Даже словарь/IDF строятся только на training folds. Все протоколы, входы и код хешировать до fitting.

## Фиксированный baseline без подбора параметров

Токены: lowercase и регулярное выражение `[a-z]+|[0-9]+`; без stopwords. Признаки: word unigram и последовательные bigram, с разными префиксами. Выбрать до2000 признаков по суммарной частоте в train, при равенстве — лексикографически. TF — обычная частота; IDF = log((1+n_train)/(1+df_train))+1. Каждый текстовый вектор нормировать L2; пустой вектор оставить нулевым.

Multi-output ridge с intercept: центрировать X и Y по training cases, alpha=1.0, без подбора. Реализация dual solve по training Gram, numpy. Predicted accuracy clip в[0,1], затем вычесть цену0.05 для каждого вопроса. Выбор максимума с tolerance1e-12, tie-order no-question→issue→product→money_gone→compensation→customer_name. Статистика всего набора не входит в fit. Имена пользователей в тексте не удаляются вручную; source IDs в признаки не входят.

## Проверка и baselines

Leave-one-component-out по case/точному видимому тексту. Предсказание и выбранная ветвь записываются до применения held-out labels в scoring. На каждой fold сохранить train/test case IDs, словарь/IDF, training counts, model matrices, raw/clipped predictions, выборы и хеши. Training cases равновесны; итоговая оценка по case 1/24.

Сравнения: no-question; каждый заранее фиксированный вопрос; uniform-random среди пяти вопросов; постоянная ветвь, выбранная только по средней training utility; ridge selector. Для training-chosen constant рассматриваются все6 ветвей, одинаковая цена/tie-order. All-data best fixed и E014 post-hoc oracle показывать только отдельными ориентирами; они не равноправные обучаемые baselines.

Метрики: accuracy, utility=accuracy−0.05×ask, ask rate, распределение выбранных вопросов, MAE шести clipped accuracy forecasts. Сохранить все held-out cases, а не только правильные. Две сводки:24case и21case после прежних exclusions00054/00131/00345. Вторая сводка только фильтрует уже полученные held-out предсказания; **не переобучает** на сокращённом наборе и не выбирает параметры. Дополнительно посчитать парный описательный разрыв с training-chosen constant; не делать p-values по24 зависимым folds.

## Верификация и решение

Отдельный код заново проверяет source labels, group disjointness, train-only vocabulary/IDF и ridge predictions через SVD вместо dual solve. Пересчитать utility/choices отдельно. Проверить, что одинаковые held-out тексты получают одно действие и что перестановка test labels не может менять предсказания (scoring изолирован от predict).

Нулевой/отрицательный результат не доказывает, что задача невозможна: мало случаев и слабые текстовые признаки. Положительный не доказывает новизну или перенос. Результат определяет, нужен ли следующий заранее разделённый набор большей мощности. Не подбирать alpha/features/семантические исключения до победы. Новых внешних отправок, API или тяжёлого обучения нет.
