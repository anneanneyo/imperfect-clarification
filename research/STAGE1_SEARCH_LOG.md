# Журнал литературного этапа

Дата доступа: 8 октября 2026. Адресный поиск с чтением первоисточников. Это не PRISMA-обзор; число найденных результатов и полнота покрытия не оценивались. Поисковые сниппеты не использовались для утверждений об отсутствии экспериментов.

## Как искали

1. Проверка ссылок и названий из исходного плана; чтение метода, экспериментальных условий и приложений о симуляторе, награде, информационном доступе.
2. Поиск прямых конкурентов по прогнозу ответа, конечной полезности, answerability, user simulation и переносу.
3. Предшественники за пределами LLM: Bayesian preference elicitation, noisy response models, model misspecification, experimental design.
4. Дополнительная проверка более узкой формулировки и свежих источников. Нерелевантная выдача не интерпретировалась как доказательство отсутствия литературы.

Использованные семейства запросов: `LLM clarification imperfect user responses expected value information response model`; `decision theoretic clarification user response model active preference elicitation`; `clarification answerability task reward`; `Bayesian preference elicitation noisy responses`; `robust Bayesian experimental design misspecification`; точные названия обнаруженных статей. Заключительные запросы: `LLM clarification "response model" "adaptation" user`; `clarification questions "distribution shift" user answerability`; `"preference elicitation" "misspecification"`. Для уточняющих запросов ограничивались arXiv, ACL Anthology, OpenReview и PMLR.

## Реестр прочитанного

Версия указана только там, где зафиксирована. Ссылки без номера версии — состояние страницы на дату доступа, не неизменяемый снимок. Перед воспроизведением нужны локальные PDF/хеши и commit кода.

| Источник | Проверенные части / глубина |
|---|---|
| [Dong, arXiv v1](https://arxiv.org/html/2601.06407v1), [PDF](https://arxiv.org/pdf/2601.06407) | §§3–7, релевантные приложения A–C; PDF prompts Fig. 10–13. Модель вероятностей, полезность, эксперименты, инструкции симулятору. |
| [ClarifyMT](https://arxiv.org/html/2512.21120), [PDF](https://arxiv.org/pdf/2512.21120) | §3.2, §§4–6, приложения с реализацией, Fig. 6–7: критерии оценки и генерации. |
| [Active Task Disambiguation](https://arxiv.org/html/2502.04485) | §2.1, метод, основные эксперименты, приложения C–E: алгоритм, настройки, неоднозначности. |
| [Rao, v2](https://arxiv.org/html/1805.04655v2), [ACL 2018](https://aclanthology.org/P18-1255/) | Основной текст §§2–5, модели, direct baseline, варианты оценки. Отдельный supplementary не получен; его чтение не заявляется. |
| [SAGE, опубликованный PDF](https://aclanthology.org/2026.findings-acl.2028.pdf), [preprint HTML](https://arxiv.org/html/2511.08798) | Метод §§4–5, алгоритм и инструкции; приложения C.1–C.5 о симуляторе и его валидации. Полный перечень API и все таблицы доменов построчно не проверялись. |
| [IG-clarifier, v1](https://arxiv.org/html/2606.03135v1) | Метод, основные эксперименты, приложения о training/evaluation, failure case и настройках. Некоторые prompts в HTML не раскрыты; полный аудит кода не выполнен. |
| [CLARITI](https://arxiv.org/html/2604.14624) | §§2–6, релевантное приложение A.8: данные, компоненты награды, ответимость и постановка SWE. |
| [UserVille / PPP](https://arxiv.org/html/2511.02208) | §§4–8, приложения A/B и E: схема пользователя, награды, перенос, human study, таблица seen/unseen предпочтений. Точный информационный доступ в реализации ещё требует проверки кода. |
| [Modeling Future Conversation Turns](https://arxiv.org/html/2410.13788) | §§2–3, §5, основные сравнения и схема формирования preferences. Не полный аудит приложений. |
| [Learning to Clarify / ACT](https://arxiv.org/html/2406.00222) | Выборочно §§3.1–3.2: будущие траектории и обучение решения уточнять. Дополнительный предшественник, не детально проверенный baseline. |
| [OPEN](https://arxiv.org/html/2403.05534) | §§2–3, §§4.3/4.5: Bayesian elicitation и пользовательские эксперименты. Не полный аудит приложений. |
| [Active Calibration, ACL 2026](https://aclanthology.org/2026.acl-long.259.pdf) | Постановка, §5.2, Algorithm 1, приложения E/G.2/G.3, релевантные prompts. Доказательства теорем не проверены. |
| [Viappiani & Boutilier 2011](https://www.cs.toronto.edu/~cebly/Papers/Viappiani_Boutilier_AAAI11.pdf) | Bayesian elicitation, определение EVOI и noisy response models; первичный авторский PDF. Не проверка всех доказательств в предшествующей работе 2010 года. |
| [Vendrov et al. 2020](https://www.cs.toronto.edu/~cebly/Papers/Vendrov_etAl_elicitation_aaai20.pdf) | §§2–3: EVOI против IG, utility, канал, posterior и myopic query selection. Метод оптимизации и результаты целиком не аудированы. |

## Дополнительный скрининг — только аннотация/начальная часть

Эти работы не дают основания для вывода «там точно нет нашей постановки»:

- [On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html): ошибочная модель и ценность информации; важное ограничение для возможных теоретических притязаний.
- [Maximin Robust Bayesian Experimental Design](https://arxiv.org/abs/2603.14094): соседняя теория устойчивого дизайна.
- [Possibilistic preference elicitation by minimax regret](https://proceedings.mlr.press/v161/adam21a.html): ошибки пользователя и misspecification.
- [Preference is More Than Comparisons](https://arxiv.org/abs/2511.09047): расширенная обратная связь и эффективность elicitation; найдено в заключительном поиске.
- [CruxBench](https://arxiv.org/abs/2609.35879): полезность вопросов для прогнозирования будущих событий; иная обратная связь, не выбран первой средой.
- [Synthetic Clarification and Correction Dialogues](https://arxiv.org/abs/2503.14167): диалоги о табличных задачах; запасной источник среды.
- [Uncertainty Decomposition](https://arxiv.org/html/2606.19559v1): исходный соседний кандидат; недостаточно для отдельного заключения о методе.

## Доступ и ограничения

PDF читались через извлечение текста web-инструментом. Скриншоты не дали доступного визуального результата; визуальная сверка формул не заявляется. Где HTML скрывал prompts, использован текст PDF. Неразборчивые формулы не служили основанием для новых математических утверждений.

Страницы репозиториев Dong, IG-clarifier, CLARITI открывались для первичной проверки наличия материалов; наличие страницы не означает доступность всех данных, лицензии, весов или успешного запуска. ICML/OpenReview-страницы IG-clarifier не позволили независимо подтвердить статус принятия; в отчёте он не утверждается. Для SAGE и Active Calibration использованы опубликованные PDF ACL.

Внешний код не запускался, зависимости и платные API не подключались. Сетевое скачивание из shell не удалось из-за DNS; web-доступ позволил завершить литературный этап. Локальных копий PDF и полностью воспроизводимого машинного поискового экспорта нет. Список выше — журнал охвата, а не обещание исчерпывающего поиска.
