# Смысловой аудит расхождений E019

8 ноября 2026. После исходов E019, до нового prompt E020. Один ассистент, без ослепления и независимой человеческой разметки.

Из всех 60 training-case с хотя бы одним расхождением взяты первые 24 по SHA256(`E019-review-v1:` + case_id). Все расхождения выбранных случаев прочитаны по полным текстам. Это целевой разбор ошибок; долю ошибок во всём наборе по нему не заявляем. Первичные scores E019 не меняются.

{"model_not_supported_by_text": 39, "model_supported_annotation_incomplete": 1, "ambiguous_text_or_ontology": 4}

| Case | Расхождения | Основание |
|---|---|---|
| 00037 | money_gone: model_not_supported_by_text | Отказ от компенсации не сообщает, было ли списание. |
| 00234 | product: model_supported_annotation_incomplete, money_gone: model_not_supported_by_text, compensation: model_not_supported_by_text | Lost card прямо называет карту; списание и компенсация не сообщены. |
| 00143 | money_gone: model_not_supported_by_text | Неизвестное списание на statement не доказывает отсутствие движения денег; отказ от компенсации относится к другому полю. |
| 00176 | money_gone: model_not_supported_by_text | Failed payment не исключает списание; факт списания не указан. |
| 00166 | product: model_not_supported_by_text, money_gone: model_not_supported_by_text | Тип продукта не назван; сбой платежа не означает отсутствие списания. |
| 00279 | issue: model_not_supported_by_text, product: model_not_supported_by_text, compensation: model_not_supported_by_text | Списание названо, но не его причина, тип продукта или запрос компенсации. |
| 00248 | issue: model_not_supported_by_text, money_gone: model_not_supported_by_text, compensation: model_not_supported_by_text | Назван только current account; выбор двух типов проблемы и отрицаний не поддержан. |
| 00340 | issue: ambiguous_text_or_ontology, product: model_not_supported_by_text | Charge that does not look right допускает неоднозначность issue; карта не названа. |
| 00379 | issue: model_not_supported_by_text | Названа карта и два отрицания; тип проблемы не назван. |
| 00144 | issue: model_not_supported_by_text, product: model_not_supported_by_text | Списание и компенсация указаны, но тип проблемы и тип счёта не названы. |
| 00181 | compensation: model_not_supported_by_text | Отсутствие списания не означает отказ от компенсации; просьба разрешить вопрос не задаёт её статус. |
| 00175 | issue: model_not_supported_by_text, product: model_not_supported_by_text, money_gone: model_not_supported_by_text, compensation: model_not_supported_by_text | Общие account/banking details не задают ни тип продукта, ни issue, ни отрицания. |
| 00490 | issue: model_not_supported_by_text, money_gone: model_not_supported_by_text, compensation: model_not_supported_by_text | Current account указан, но общий вопрос не равен change_request; отрицания не сообщены. |
| 00334 | compensation: ambiguous_text_or_ontology | Ask if I can get money back as compensation можно читать как просьбу или как вопрос о возможности; сохраняем спорность вместо объявления однозначной ошибки. |
| 00079 | money_gone: model_not_supported_by_text | Непризнанная операция и отказ от компенсации не доказывают money_gone=no. |
| 00026 | issue: model_not_supported_by_text | Lost card явно указан; добавление failed_payment не поддержано текстом. |
| 00425 | issue: model_not_supported_by_text | Savings и два отрицания названы; вопрос о комиссии не сообщён. |
| 00315 | money_gone: model_not_supported_by_text | Отказ от компенсации не сообщает отсутствие списания. |
| 00034 | product: model_not_supported_by_text, compensation: model_not_supported_by_text | No money left указано; generic account не задаёт current_account, компенсация не обсуждается. |
| 00268 | compensation: model_not_supported_by_text | Несанкционированное списание указано, но отказ от компенсации не сообщён. |
| 00116 | issue: ambiguous_text_or_ontology, product: model_not_supported_by_text, money_gone: model_not_supported_by_text, compensation: model_not_supported_by_text | Login error выходит за перечисленные issue-категории; failed_payment не следует из текста. Остальные конкретизации также не сообщены. |
| 00293 | product: model_not_supported_by_text | Непризнанная операция на account не задаёт продукт card. |
| 00467 | issue: ambiguous_text_or_ontology, product: model_not_supported_by_text | Fee/charge is not right допускает спорную границу issue; current_account не назван. |
| 00214 | issue: model_not_supported_by_text, compensation: model_not_supported_by_text | Savings и отсутствие списания названы; комиссия и отказ от компенсации не сообщены. |

Контроль E020 должен требовать текстовое свидетельство при сужении множества значений. Наличие точной цитаты само по себе не доказывает смысловую поддержку. Цитата и её содержание требуют отдельных проверок. Не объявлять такое извлечение новым научным методом.

[Выборка с полными текстами](../results/e019_semantic_review/selection.json), [разметка](../results/e019_semantic_review/labels.json), [хеши](../results/e019_semantic_review/manifest.json).
