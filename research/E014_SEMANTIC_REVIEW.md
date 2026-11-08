# E014: проверка текстов до модельных исходов

27 октября 2026. Один ассистент; это не независимая человеческая разметка.

Прочитаны все 187 уникальных сочетаний текст/вопрос/ожидаемые факты. Supported:182; unclear:5; явных противоречий:0. Структурные проверки отдельно подтверждают gold по schema и совместимость reveal с полным профилем.

## Отмеченные неоднозначности

### Единица 63: initial

Initial text adds a failed-login issue outside the schema while evidence is empty. This can influence interpretation; no explicit contradiction with any positive evidence.

Source IDs: train/banking_support/00131/p0/a

### Единица 67: compensation

Answer reverses speaker perspective (you asking for compensation). Both binary values remain possible; no usable new constraint, but wording is unclear.

Source IDs: train/banking_support/00131/p0/a

### Единица 161: money_gone

Answer refers to your account from a user turn. Both binary values remain possible; perspective is unclear.

Source IDs: train/banking_support/00345/p1/a

### Единица 173: initial

Explicit fee-question framing coexists with unauthorized deduction. Fee dispute versus unrecognized charge is semantically ambiguous under exclusive issue labels.

Source IDs: train/banking_support/00054/p0/a, train/banking_support/00054/p1/a

### Единица 174: issue

Repeats fee-question framing and unauthorized deduction; same ambiguity as the initial message.

Source IDs: train/banking_support/00054/p0/a, train/banking_support/00054/p1/a

## Решение до генерации

Основной прогон включает все 24 исходные case-группы. Заранее заданная чувствительность убирает целиком три группы: train/banking_support/00054, train/banking_support/00131, train/banking_support/00345. Ни ответы, ни gold не переписываются. Дальнейшие оценки ошибок исполнителя на этих группах требуют оговорки об интерпретации текста.

Стилистические огрехи, не меняющие решающую информацию (например обращение к собственному имени), отмечены отдельно и не являются основанием для исключения. Ответ с двумя возможными значениями бинарного признака может быть информационно эквивалентен «не знаю»; метка partial не означает положительную пользу.

[Все исходные единицы](../results/lavoir_executor_e014/review_units.json), [сохранённые метки](../results/lavoir_executor_e014/semantic_review.json).
