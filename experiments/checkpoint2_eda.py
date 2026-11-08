"""Checkpoint 2: deterministic derived dataset, EDA and provenance (no model calls).

Reads saved E002 outputs; --repo optionally verifies the pinned upstream source.
No raw upstream catalogs, preference vectors or utterances are redistributed.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/learn_question_value"
COMMIT = "27466a7832d5aafff82017a659e08942e18b01ae"
SPLITS = ["fit", "validation", "holdout"]
SPLIT_NAMES = {"fit": "Обучение", "validation": "Настройка", "holdout": "Разработка / holdout"}
SLUGS = ["price", "stops", "layover", "arrival", "american", "delta", "jetblue", "southwest"]
QNAMES = ["Цена", "Пересадки", "Ожидание", "Прибытие", "American", "Delta", "JetBlue", "Southwest"]
INDICES = [7, 6, 5, 0, 1, 2, 3, 4]
COLORS = ["#267c9c", "#e69b34", "#8774b1"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def stats(values):
    a = np.array(values, dtype=float)
    return {"n": len(a), "min": float(a.min()), "q25": float(np.quantile(a, .25)),
            "median": float(np.median(a)), "mean": float(a.mean()),
            "q75": float(np.quantile(a, .75)), "max": float(a.max()),
            "zero_within_1e-12": int((np.abs(a) <= 1e-12).sum())}


def catalog_hash(options):
    return hashlib.sha256(json.dumps(sorted(options), sort_keys=True).encode()).hexdigest()


def source_audit(repo, tasks, summary):
    if repo is None:
        return {"status": "not_requested", "note": "Полный аудит источника требует --repo; E002-файлы проверены отдельно."}
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    assert commit == COMMIT, "Upstream commit differs"
    base = repo / "flight_rec/data"
    raw = {}
    for name in ["train.json", "eval.json"]:
        assert sha(base / name) == summary["input_sha256"][name], name
        raw[name] = json.loads((base / name).read_text())
    profiles, audit = {}, {}
    prior = np.array([raw["train.json"][t["id"]]["reward_function"] for t in tasks["fit"]])
    max_label_error = 0.
    for split, rows in tasks.items():
        source = raw["eval.json" if split == "holdout" else "train.json"]
        profiles[split] = {tuple(source[t["id"]]["reward_function"]) for t in rows}
        for task in rows:
            record = source[task["id"]]
            options = np.array(record["rounds"][-1]["options"])
            assert catalog_hash(record["rounds"][-1]["options"]) == task["catalog_hash"]
            assert options.shape == (3, 8) and np.isfinite(options).all()
            assert (options >= 0).all() and (options <= 1).all()
            assert np.array(record["reward_function"]).shape == (8,)
            scores = prior @ options.T
            initial = scores.mean(axis=0).max()
            risk = scores.max(axis=1).mean()-initial
            max_label_error = max(max_label_error, abs(float(risk)-task["risk"]))
            for q,index in enumerate(INDICES):
                signs = np.sign(prior[:,index]).astype(int)
                span = float(np.ptp(options[:,index]))
                e = [.4,.7,.5,.75,.6,.6,.6,.6][q]-.15*span
                b = .2+.1*span
                assert np.allclose(task["rates"][q],[e,b],atol=1e-12,rtol=0)
                # Explicit sum of joint utilities over visible outcomes; no sampled replies.
                full_total = sum((scores*(signs==v)[:,None]).mean(0).max() for v in [-1,0,1])
                partial_total = sum((scores*np.isin(signs,pair)[:,None]/2).mean(0).max()
                                    for pair in [(-1,0),(-1,1),(0,1)])
                ideal = max(0.,float(full_total-initial))
                actual = max(0.,float(e*full_total+b*partial_total+(1-e-b)*initial-initial))
                discarded = max(0.,float(e*(full_total-initial)))
                max_label_error = max(max_label_error, abs(ideal-task["ideal"][q]),
                                      abs(actual-task["target"][q]),abs(discarded-task["discard"][q]))
        audit[split] = {"tasks": len(rows), "distinct_profile_vectors": len(profiles[split])}
    overlaps = {a + "__" + b: len(profiles[a] & profiles[b])
                for i, a in enumerate(SPLITS) for b in SPLITS[i+1:]}
    assert overlaps["fit__validation"] == 0
    assert max_label_error < 1e-12
    # An eval preference vector may match training: these labels use only the fit prior,
    # never the eval record's preference. Do not mislabel this as unseen-user evaluation.
    licenses = sorted(str(p.relative_to(repo)) for p in repo.rglob("*")
                      if p.is_file() and p.name.lower() in {"license", "license.txt", "license.md", "copying"})
    return {"status": "passed_with_declared_profile_overlap", "commit": commit,
            "raw_input_sha256": {n: sha(base / n) for n in raw},
            "raw_record_counts": {n: len(v) for n, v in raw.items()},
            "source_round_counts": {n: dict(sorted(Counter(len(r["rounds"]) for r in v).items())) for n, v in raw.items()},
            "selected_catalog_shape": [3, 8], "all_selected_features_finite_and_in_0_1": True,
            "all_6496_labels_recomputed_from_source": True, "max_recomputed_label_error": max_label_error,
            "profiles_by_split": audit, "profile_vector_overlap": overlaps,
            "standalone_license_files_found": licenses,
            "interpretation": "Совпадение одного профиля fit/holdout не является утечкой eval-целей: prior строится только по fit. Нельзя называть holdout полностью новыми пользователями."}


def flatten(tasks):
    rows = []
    for split in SPLITS:
        for t in tasks[split]:
            scope = "eval" if split == "holdout" else "train"
            task_id = "%s:%d" % (scope, t["id"])
            for q in range(8):
                block = t["x"][q][q*8:q*8+8]
                exact, partial = t["rates"][q]
                rows.append({"schema_version": "e002-derived-v1", "task_id": task_id,
                    "source_partition": scope, "record_index": t["id"],
                    "analysis_split": "development_holdout" if split == "holdout" else split,
                    "catalog_sha256": t["catalog_hash"], "question_id": q,
                    "question": SLUGS[q], "upstream_feature_index": INDICES[q],
                    "feature_span": block[2], "mean_reward_margin": block[5],
                    "catalog_feature_span_l2": block[6],
                    "ideal_sign_gain": t["ideal"][q],
                    "expected_imperfect_gain": t["target"][q],
                    "expected_gain_discard_partial": t["discard"][q],
                    "no_question_risk": t["risk"],
                    "p_exact": exact, "p_partial": partial, "p_unknown": 1-exact-partial,
                    "ridge_active_feature_block": block})
    return rows


def validate(tasks, rows, splits, summary, per_task, model):
    assert {s: len(ts) for s, ts in tasks.items()} == summary["split_counts"]
    assert len(rows) == 6496
    assert len({(r["task_id"], r["question_id"]) for r in rows}) == len(rows)
    assert len({t["catalog_hash"] for ts in tasks.values() for t in ts}) == 812
    groups = [{t["catalog_hash"] for t in tasks[s]} for s in SPLITS]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i+1,3))
    for s in SPLITS:
        assert [(t["id"],t["catalog_hash"]) for t in tasks[s]] == [(t["record"],t["hash"]) for t in splits["tasks"][s]]
    assert {t["id"] for t in tasks["fit"]} == set(splits["prior_fit_record_ids"])
    assert not {t["id"] for t in tasks["fit"]} & {t["id"] for t in tasks["validation"]}
    nulls = {key: sum(r[key] is None for r in rows) for key in rows[0]}
    nonfinite = 0
    max_feature_error = 0.
    for r in rows:
        numeric = [v for v in r.values() if isinstance(v, (int,float))] + r["ridge_active_feature_block"]
        nonfinite += int(not np.isfinite(numeric).all())
        p = [r["p_exact"], r["p_partial"], r["p_unknown"]]
        assert min(p) >= 0 and max(p) <= 1 and abs(sum(p)-1) < 1e-12
        assert -.0000000001 <= r["expected_gain_discard_partial"] <= r["expected_imperfect_gain"]+1e-10
        assert r["expected_imperfect_gain"] <= r["ideal_sign_gain"]+1e-10 <= r["no_question_risk"]+2e-10
        i, s, m = r["ideal_sign_gain"], r["feature_span"], r["mean_reward_margin"]
        block = [1, i, s, i*s, i*s*s, m, r["catalog_feature_span_l2"], i*m]
        max_feature_error = max(max_feature_error, float(np.max(np.abs(np.array(block)-r["ridge_active_feature_block"]))))
    assert not nonfinite and not sum(nulls.values()) and max_feature_error < 1e-12
    holdout = tasks["holdout"]
    h = np.concatenate([t["x"] for t in holdout])
    ideal = np.array([t["ideal"] for t in holdout])
    actual = np.array([t["target"] for t in holdout])
    risk = np.array([t["risk"] for t in holdout])
    z = np.column_stack([np.ones(len(h)), (h-model["center"])/model["scale"]])
    preds = {"ideal": ideal, "calibrated_simple": ideal*model["slopes"],
             "ridge": np.clip((z@model["weights"]).reshape(-1,8), 0, ideal), "oracle": actual,
             "oracle_discard_ambiguity": np.array([t["discard"] for t in holdout])}
    max_metric_error = 0.
    for metric in summary["metrics"]:
        name, cost = metric["method"], metric["cost"]
        if name == "no_ask": losses = risk
        elif name == "random": losses = risk-actual.mean(1)+cost
        else:
            p = preds[name]; q = p.argmax(1); ask = np.ones(len(p),bool) if cost == 0 else p.max(1)>cost
            truth = preds[name] if name == "oracle_discard_ambiguity" else actual
            losses = risk-truth[np.arange(len(p)),q]*ask+cost*ask
            original = [r for r in per_task if r["method"] == name and r["cost"] == cost]
            assert len(original) == 90
            assert all(r["task"] == holdout[i]["id"] and r["question"] == (int(q[i]) if ask[i] else None)
                       and abs(r["loss"]-losses[i]) < 1e-12 for i,r in enumerate(original))
        max_metric_error = max(max_metric_error, abs(float(losses.mean())-metric["loss"]))
    assert max_metric_error < 1e-12
    return {"all_passed": True, "missing_values_by_field": nulls, "nonfinite_rows": nonfinite,
            "duplicate_task_question_ids": 0, "catalog_overlap_between_splits": 0,
            "all_eight_questions_stay_in_same_split": True, "fit_prior_record_ids_match": True,
            "train_fit_validation_record_id_overlap": 0,
            "raw_numeric_record_id_scope_warning": "train:0 и eval:0 — разные записи; ключ обязан включать source_partition.",
            "max_reconstructed_feature_error": max_feature_error, "max_replayed_metric_error": max_metric_error,
            "saved_model_and_per_task_replayed": True}, preds


def analysis(tasks, rows, checks, upstream, summary, preds):
    per_split = {}
    for s in SPLITS:
        ts = tasks[s]; full = np.array([t["ideal"] for t in ts]); actual = np.array([t["target"] for t in ts])
        risk = np.array([t["risk"] for t in ts]); rates = np.array([t["rates"] for t in ts])
        per_split[s] = {"tasks": len(ts), "question_rows": len(ts)*8,
            "no_question_risk": stats(risk), "ideal_sign_gain": stats(full.ravel()),
            "expected_imperfect_gain": stats(actual.ravel()),
            "best_actual_question_counts": np.bincount(actual.argmax(1), minlength=8).tolist(),
            "mean_best_ideal_gain": float(full.max(1).mean()), "mean_best_actual_gain": float(actual.max(1).mean()),
            "regret_zero_questions": float(risk.mean()),
            "regret_one_ideal_sign_question": float((risk-full.max(1)).mean()),
            "regret_one_imperfect_question_oracle": float((risk-actual.max(1)).mean()),
            "regret_one_random_imperfect_question": float((risk-actual.mean(1)).mean()),
            "all_tasks_have_positive_best_imperfect_gain": bool((actual.max(1)>1e-12).all())}
    qstats = []
    for q in range(8):
        rs = [r for r in rows if r["question_id"] == q]
        qstats.append({"question_id": q, "name": SLUGS[q], "label": QNAMES[q], "n": len(rs),
            "expected_imperfect_gain": stats([r["expected_imperfect_gain"] for r in rs]),
            "ideal_sign_gain": stats([r["ideal_sign_gain"] for r in rs]),
            "feature_span": stats([r["feature_span"] for r in rs]),
            "p_exact": stats([r["p_exact"] for r in rs]), "p_partial": stats([r["p_partial"] for r in rs]),
            "p_unknown": stats([r["p_unknown"] for r in rs])})
    counts = {name: np.bincount(p.argmax(1), minlength=8).tolist() for name,p in preds.items()}
    oracle_q = preds["oracle"].argmax(1)
    agreement = {name: int((p.argmax(1)==oracle_q).sum()) for name,p in preds.items()}
    return {"artifact_version": "checkpoint2-2026-10-27-v1", "status": "derived_structured_synthetic_eda",
        "unit_of_analysis": "812 каталогов; 6496 зависимых строк задача–вопрос, не 6496 независимых пользователей",
        "questions": [{"id":q,"slug":SLUGS[q],"label":QNAMES[q]} for q in range(8)],
        "total_tasks": 812, "total_question_rows": len(rows), "per_split": per_split, "per_question": qstats,
        "quality_checks": checks, "upstream_audit": upstream, "holdout_question_choices": counts,
        "holdout_question_agreement_with_oracle": agreement, "e002_metrics": summary["metrics"],
        "e002_paired_contrasts": summary["paired_contrasts"],
        "caveats": ["Механизм ответов задан исследователем; это не частоты ответов людей.",
                    "Точные целевые значения интегрированы по prior; E002 имеет привилегированную симуляционную разметку.",
                    "Признак ideal_sign_gain также аналитический; модель не обучается понимать текст.",
                    "Разработка и анализ многократно использовали 90 holdout-задач; это development-набор, не финальный тест.",
                    "Все задачи имеют один домен, три варианта и один вопрос; устройства, курсы, два вопроса и H3 ещё не проверены.",
                    "Отдельный файл лицензии в pinned upstream не обнаружен; raw данные не включены, разрешение на публичное распространение производных данных не заявлено."]}


def plot_all(out, a, tasks, preds):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.titlesize":12,
                        "axes.labelsize":10, "axes.spines.top":False, "axes.spines.right":False,
                        "figure.facecolor":"white", "svg.hashsalt":"checkpoint2-2026-10-27"})
    saved = []
    def save(fig, name):
        fig.savefig(out/(name+".png"), dpi=160, bbox_inches="tight", metadata={"Software":"checkpoint2_eda"})
        fig.savefig(out/(name+".svg"), bbox_inches="tight", metadata={"Date":None, "Creator":"checkpoint2_eda"})
        plt.close(fig); saved.append(name)
    fig, axes = plt.subplots(1,2,figsize=(9.2,4.3))
    for ax,key,title in zip(axes,["tasks","question_rows"],["812 каталожных задач","6 496 строк «задача–вопрос»"]):
        vals = [a["per_split"][s][key] for s in SPLITS]
        ax.bar(range(3),vals,color=COLORS,width=.65)
        ax.set_xticks(range(3),["Обучение","Настройка","Разработка"],rotation=10)
        ax.set_title(title);ax.set_ylim(0,max(vals)*1.2);ax.set_ylabel("Количество")
        for x,v in enumerate(vals): ax.text(x,v+max(vals)*.025,f"{v:,}".replace(","," "),ha="center")
    fig.suptitle("Одна задача даёт восемь связанных строк",fontsize=14)
    fig.tight_layout();save(fig,"01_composition")
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.3))
    for s,c in zip(SPLITS,COLORS):
        risk = [t["risk"] for t in tasks[s]]
        gain = np.array([t["target"] for t in tasks[s]]).ravel()
        axes[0].hist(risk,bins=np.linspace(0,1.15,24),density=True,histtype="step",lw=2,color=c,label=SPLIT_NAMES[s])
        axes[1].hist(gain,bins=np.linspace(0,.45,24),density=True,histtype="step",lw=2,color=c)
    axes[0].set(xlabel="Ожидаемая потеря без вопроса",ylabel="Плотность",title="Трудность исходного выбора")
    axes[1].set(xlabel="Ожидаемое уменьшение потери",ylabel="Плотность",title="Ожидаемая польза вопроса")
    axes[0].legend(fontsize=8)
    fig.suptitle("Распределения близки; это один искусственный режим",fontsize=13)
    fig.tight_layout();save(fig,"02_risk_and_gain")
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.3),gridspec_kw={"width_ratios":[1,1.2]})
    full=np.array([t["ideal"] for ts in tasks.values() for t in ts]);actual=np.array([t["target"] for ts in tasks.values() for t in ts])
    axes[0].scatter(full.ravel(),actual.ravel(),s=5,alpha=.23,color=COLORS[0],rasterized=True)
    bound=float(full.max());axes[0].plot([0,bound],[0,bound],"--",color="#777777",lw=1,label="Без потери информации")
    axes[0].set(xlabel="Польза точного знака предпочтения",ylabel="Польза несовершенного ответа",title="Каждая точка — задача и вопрос")
    axes[0].legend(fontsize=7)
    ratios=[actual[:,q][full[:,q]>1e-12]/full[:,q][full[:,q]>1e-12] for q in range(8)]
    bp=axes[1].boxplot(ratios,vert=False,tick_labels=QNAMES,showfliers=False,patch_artist=True)
    for patch in bp["boxes"]:patch.set_facecolor("#aed4dd")
    axes[1].set(xlabel="Доля сохранённой пользы",xlim=(0,1),title="Разные вопросы теряют разную долю")
    fig.tight_layout();save(fig,"03_ideal_and_imperfect")
    fig,ax=plt.subplots(figsize=(9.2,4.3))
    y=np.arange(8);left=np.zeros(8)
    for key,label,c in zip(["p_exact","p_partial","p_unknown"],["Точный знак","Два возможных знака","Не знаю"],COLORS):
        vals=np.array([d[key]["mean"] for d in a["per_question"]]);ax.barh(y,vals,left=left,label=label,color=c)
        for j,v in enumerate(vals):ax.text(left[j]+v/2,j,f"{v:.1%}",ha="center",va="center",color="white",fontsize=9)
        left+=vals
    ax.set_yticks(y,QNAMES);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel("Средняя заданная вероятность по 812 каталогам")
    ax.set_title("Состав ответов задаёт симулятор — это не измерение поведения людей")
    ax.legend(loc="lower center",bbox_to_anchor=(.5,-.27),ncol=3,fontsize=9)
    fig.tight_layout();save(fig,"04_synthetic_answer_types")
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.5),gridspec_kw={"width_ratios":[1,1.1]})
    h=a["per_split"]["holdout"];start=h["regret_zero_questions"]
    for key,label,c in [("regret_one_ideal_sign_question","Всегда точный знак",COLORS[0]),
                        ("regret_one_imperfect_question_oracle","Неполные ответы: oracle",COLORS[1]),
                        ("regret_one_random_imperfect_question","Неполные: случайный вопрос",COLORS[2])]:
        axes[0].plot([0,1],[start,h[key]],marker="o",lw=2,color=c,label=label)
        axes[0].text(1.03,h[key],f"{h[key]:.3f}",fontsize=9,color=c,va="center")
    axes[0].set(xlabel="Число вопросов",ylabel="Средняя потеря ↓",title="90 задач разработки; цена вопроса = 0",xlim=(-.05,1.24),xticks=[0,1],ylim=(.27,.65))
    axes[0].legend(fontsize=7,loc="upper right")
    methodnames=[("ideal","Предполагает точный ответ"),("calibrated_simple","Простая поправка"),("ridge","Регрессия"),("oracle","Oracle")]
    for i,(m,l) in enumerate(methodnames):
        counts=np.array(a["holdout_question_choices"][m]);axes[1].barh(np.arange(8)+(i-1.5)*.19,counts,height=.18,label=l)
    axes[1].set_yticks(np.arange(8),QNAMES);axes[1].invert_yaxis();axes[1].set(xlabel="Сколько раз выбран вопрос",title="Какие вопросы выбирают методы")
    axes[1].legend(fontsize=7,loc="lower right")
    fig.tight_layout();save(fig,"05_questions_and_loss")
    return saved


def documents(out, a):
    field_table = [
      ("schema_version","string","Версия производного формата."),
      ("task_id","string","Уникальный ключ source_partition:record_index; все 8 строк одной задачи имеют общий ключ."),
      ("source_partition; record_index","string; int","train/eval исходного проекта и индекс записи; индекс отдельно не уникален."),
      ("analysis_split","string","fit / validation / development_holdout; последний уже использовался при разработке."),
      ("catalog_sha256","string","Хеш отсортированных альтернатив, позволяет обнаружить перестановочные дубли."),
      ("question_id; question; upstream_feature_index","int; string; int","Номер вопроса 0–7, его тип и индекс исходной характеристики."),
      ("feature_span","float","Размах соответствующей характеристики по трём альтернативам, [0,1]."),
      ("mean_reward_margin","float","Разность двух лучших ожидаемых полезностей под fit-prior, ≥0."),
      ("catalog_feature_span_l2","float","Евклидова норма вектора размахов восьми характеристик."),
      ("ideal_sign_gain","float","Аналитическая польза раскрытия точного знака коэффициента; привилегированный структурный признак E002, не наблюдение человека."),
      ("expected_imperfect_gain","float","Целевая ожидаемая польза при синтетическом канале; вычислена точно, не измерена в реальном диалоге."),
      ("expected_gain_discard_partial","float","Аналитическая польза, если неоднозначные ответы интерпретировать как отсутствие информации."),
      ("no_question_risk","float","Ожидаемая потеря без уточнения относительно знающего весь профиль решения."),
      ("p_exact; p_partial; p_unknown","float","Истинные вероятности симулятора, сумма=1; это служебные поля, НЕ вход регрессии E002."),
      ("ridge_active_feature_block","array[8]","[1, ideal, span, ideal×span, ideal×span², margin, spread, ideal×margin]. Остальные блоки полного 64-мерного вектора нулевые."),
    ]
    table="\n".join("| `%s` | %s | %s |" % r for r in field_table)
    (out/"DATA_CARD.md").write_text(f"""# Данные v1: уточнения при выборе рейса

Дата: 27 октября 2026. Формат: `data/question_values_v1.jsonl`, UTF-8, одна JSON-строка на пару задача–вопрос. **812 каталогов, 6 496 связанных строк.** Это производная структурная версия данных для учебного разведочного анализа, не сбор ответов людей и не новый независимо собранный датасет.

## Источник и преобразование

Использован открыто доступный [репозиторий VOI_communication](https://github.com/dong-river/VOI_communication/tree/{COMMIT}/flight_rec), commit `{COMMIT}`; файлы `flight_rec/data/train.json` (722 записи) и `eval.json` (90). Из каждой записи взят последний каталог: три альтернативы с восемью нормированными характеристиками. Исходные реплики, каталоги и векторы предпочтений в пакет **не скопированы**. Указанные числа не означают количество реальных участников: происхождение каталожных данных не даёт основания называть наш набор человеческим исследованием.

В E002 группы одинаковых векторов предпочтений в train делятся с seed=20261019: 577 записей обучения (576 уникальных векторов), 145 записей настройки (145 векторов). Распределение предпочтений — равномерное по 577 **записям** обучения; повторяющийся профиль получает вес двух записей. Это эмпирический prior, а не истинная популяция пользователей. Все 90 eval-каталогов используются как набор разработки. Каталоги канонизированы сортировкой альтернатив; совпадений между выборками нет. Одна пара fit/eval имеет одинаковый вектор предпочтений; eval-профили не участвуют в вычислении целевых значений. Поэтому это допустимый контроль на новых каталогах, но не утверждение о новых предпочтениях пользователей.

Исходная E002 не использует историю диалога, исходную реплику и фактический профиль конкретного eval-пользователя. Каждый каталог проверяется по одному общему fit-prior. Метки вычисляются точным перебором предпочтений и ответов, без выборочного шума. Это намеренно облегчённая диагностическая среда.

## Полные и неполные ответы

Каждый из восьми вопросов касается **знака одного коэффициента предпочтения**: цена, число пересадок, ожидание пересадки, прибытие, American, Delta, JetBlue, Southwest. «Полный» означает точный знак по этому вопросу, а не весь профиль пользователя. Частичный ответ — два возможных знака из `{{−, 0, +}}`, обязательно содержащие истинный. Например, «для меня цена либо нейтральна, либо чем ниже, тем лучше» соответствует `{{−,0}}`. «Не знаю» не сужает распределение в E002. Отдельное молчание не моделируется.

Пусть `span` — размах характеристики в каталоге. Задано `p_exact = base[q] − 0.15×span`, `p_partial = 0.2 + 0.1×span`, `p_unknown = 1 − p_exact − p_partial`. Константы base: `[0.4,0.7,0.5,0.75,0.6,0.6,0.6,0.6]`. При частичном ответе две содержащие истину пары равновероятны. При фиксированных каталоге и вопросе **тип ответа не зависит от скрытого предпочтения**. Эти вероятности назначены исследователем; столбцы и график не показывают наблюдённые частоты людей.

## Словарь полей

| Поле | Тип | Значение и роль |
|---|---|---|
{table}

Обучению разрешены `ridge_active_feature_block` (или его восстанавливаемый 64-мерный вариант) и целевая метка `expected_imperfect_gain` только на fit. Поля `p_*`, `no_question_risk`, `expected_gain_discard_partial` — для проверки и анализа. Нельзя случайно подать их регрессору и назвать результат сопоставимым с E002. `ideal_sign_gain` доступен по аналитической модели, поэтому это не оцениватель, работающий только с текстом.

## Качество и ограничения

Нет пропусков, NaN/Inf и повторов ключа (task_id,question_id); везде восемь вопросов, нормированные вероятности, допустимый порядок ожидаемых выгод. Нулевые выгоды существуют и содержательны, а не являются пропусками. Разделение нужно делать по `task_id`, а не по строкам: случайное разбиение 6 496 строк создаст утечку одной задачи между train и test. Набор разработки уже многократно изучен; он не даёт честного финального теста новой гипотезы.

Один домен (рейсы), три альтернативы, максимум один вопрос. Устройства и курсы, второе уточнение, язык, реальные люди и основная гипотеза о переносе раздельной модели ещё не покрыты. Числа — в искусственных единицах полезности, не рубли и не доля успеха.

В pinned-версии внешнего репозитория отдельный файл лицензии не обнаружен. Пакет содержит только наши производные признаки, метки, агрегаты и код; это **не заявление о разрешении публично перелицензировать или распространять исходный/производный набор**. Для публичного выпуска условия требуется выяснить отдельно. `data/manifest.json` фиксирует происхождение и хеши. Пакет предназначен для локального воспроизведения и рассмотрения учебного анализа.
""")
    (out/"REPRODUCE.md").write_text("""# Воспроизведение разведочного анализа

Из корня проекта (Python 3.9+, NumPy, Matplotlib уже были доступны; сетевые вызовы и языковые модели не нужны для сборки):

```bash
python3 experiments/checkpoint2_eda.py
```

Входы: сохранённые `results/learn_question_value/{training_data.json,splits.json,summary.json,per_task.json,model.npz}`. Скрипт не обучает новую модель, а воспроизводит признаки и все сохранённые метрики E002, затем создаёт производный JSONL и пять пар PNG/SVG. Никаких случайных выборок для графиков или анализа не используется. SVG фиксирует источник случайности идентификаторов и не содержит текущую дату.

Для расширенной проверки исходных каталогов и профилей клонируйте общедоступный репозиторий вне сдаваемой папки и зафиксируйте версию:

```bash
git clone https://github.com/dong-river/VOI_communication.git /tmp/voi-checkpoint2-source
git -C /tmp/voi-checkpoint2-source checkout 27466a7832d5aafff82017a659e08942e18b01ae
python3 experiments/checkpoint2_eda.py --repo /tmp/voi-checkpoint2-source
```

`--output /tmp/checkpoint2-repeat` сохраняет результат в отдельную папку. Идентичные входы, версии NumPy/Matplotlib и одинаковый режим `--repo` дают идентичные JSON, JSONL, PNG, SVG и Markdown. Без `--repo` содержимое данных и графиков то же, но в сводке источник помечен как не проверенный в этом запуске. `analysis.json` и `data/manifest.json` содержат результаты проверок и хеши; сырые внешние данные не копируются.

Проверки: числа задач/строк, уникальные составные ключи, отсутствие совпадающих каталогов между выборками, все вопросы задачи в одной выборке, соответствие prior обучающим записям, отсутствие пропусков/NaN/Inf, нормировка каналов, границы ожидаемой пользы, восстановление 64-мерных признаков, повторный расчёт сохранённой модели, всех метрик и пообъектных результатов. С `--repo`: commit и SHA256 исходных файлов, форма/границы каталогов, хеши каталогов, группировка профилей.

Линейка «0–1 вопрос» на пятом рисунке — точный анализ уже имеющихся меток. Точки «всегда точный знак» и «неполный ответ» используют разные каналы; это не прямое доказательство преимущества алгоритма и не результат с двумя вопросами. Подписи oracle обозначают недоступное на практике знание модели ответов.
""")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--repo",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/checkpoint2_2026-10-27")
    args=parser.parse_args();out=args.output.resolve();(out/"data").mkdir(parents=True,exist_ok=True);(out/"figures").mkdir(exist_ok=True)
    tasks=json.loads((SOURCE/"training_data.json").read_text());splits=json.loads((SOURCE/"splits.json").read_text())
    summary=json.loads((SOURCE/"summary.json").read_text());per_task=json.loads((SOURCE/"per_task.json").read_text())
    with np.load(SOURCE/"model.npz") as saved: model={k:saved[k] for k in saved.files}
    rows=flatten(tasks);checks,preds=validate(tasks,rows,splits,summary,per_task,model)
    upstream=source_audit(args.repo.resolve() if args.repo else None,tasks,summary)
    a=analysis(tasks,rows,checks,upstream,summary,preds)
    target=out/"data/question_values_v1.jsonl"
    target.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n" for r in rows))
    figures=plot_all(out/"figures",a,tasks,preds);a["figures"]=figures
    a["runtime"]={"numpy":np.__version__,"matplotlib":matplotlib.__version__}
    dump(out/"analysis.json",a);documents(out,a)
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [SOURCE/n for n in ["training_data.json","splits.json","summary.json","per_task.json","model.npz"]]}
    files=[target,out/"analysis.json",out/"DATA_CARD.md",out/"REPRODUCE.md"]+[out/"figures"/(n+ext) for n in figures for ext in [".png",".svg"]]
    dump(out/"data/manifest.json",{"schema":"checkpoint2-manifest-v1","dataset_version":"e002-derived-v1",
        "upstream_url":"https://github.com/dong-river/VOI_communication","upstream_commit":COMMIT,
        "upstream_raw_sha256":summary["input_sha256"],"input_sha256":inputs,
        "generator":"experiments/checkpoint2_eda.py","generator_sha256":sha(Path(__file__)),
        "output_sha256":{str(p.relative_to(out)):sha(p) for p in files},"rows":len(rows),"tasks":812,
        "license_note":"Standalone upstream license not found; no redistribution permission asserted. Raw upstream data excluded.",
        "no_llm_calls":True,"no_human_responses":True,"source_audit_status":upstream["status"]})
    print(json.dumps({"output":str(out),"tasks":812,"rows":len(rows),"checks":checks["all_passed"],
                      "figures":figures,"upstream_audit":upstream["status"]},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
