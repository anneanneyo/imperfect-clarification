"""Exact, deliberately constructed diagnostic; not an LLM experiment.

Run: python3 experiments/clarification_sanity.py
No dependencies, sampling, training, or external API calls.
"""

import json
import math
from pathlib import Path


def risk(prior, losses):
    return min(sum(p * loss for p, loss in zip(prior, row)) for row in losses)


def entropy(prior):
    return -sum(p * math.log2(p) for p in prior if p > 0)


def channel(question, answer_probability):
    """P(response | state). Missingness is independent of the state here."""
    rows = []
    for z in range(4):
        value = {"exact": z, "important": z // 2, "minor": z % 2}[question]
        rows.append({str(value): answer_probability, "unknown": 1 - answer_probability})
    return rows


def posterior_branches(prior, response_channel):
    for answer in sorted(set().union(*(row.keys() for row in response_channel))):
        joint = [p * row.get(answer, 0) for p, row in zip(prior, response_channel)]
        probability = sum(joint)
        if probability > 0:
            yield probability, [x / probability for x in joint]


def evaluate(prior, losses, response_channel):
    branches = list(posterior_branches(prior, response_channel))
    posterior_risk = sum(p * risk(b, losses) for p, b in branches)
    information_gain = entropy(prior) - sum(p * entropy(b) for p, b in branches)
    return {"loss": posterior_risk, "value": risk(prior, losses) - posterior_risk,
            "information_gain_bits": information_gain}


def run():
    prior = [0.25] * 4
    # Four possible goals and decisions: two binary preferences.
    losses = [[0.9 * (a // 2 != z // 2) + 0.1 * (a % 2 != z % 2)
               for z in range(4)] for a in range(4)]
    questions = ["exact", "important", "minor"]
    ideal = {q: evaluate(prior, losses, channel(q, 1)) for q in questions}
    ideal_question = max(questions, key=lambda q: ideal[q]["value"])
    rows = []
    for p in [0, 0.2, 0.5, 0.8, 0.9, 1]:
        # Only the exact question changes its response rate across conditions.
        rates = {"exact": p, "important": 0.9, "minor": 1.0}
        actual = {q: evaluate(prior, losses, channel(q, rates[q])) for q in questions}
        selections = {
            "no_question": None,
            "ideal_answer_selector": ideal_question,
            "information_gain_known_channel": max(questions, key=lambda q: actual[q]["information_gain_bits"]),
            "bayesian_evsi_known_channel": max(questions, key=lambda q: actual[q]["value"]),
        }
        for method, q in selections.items():
            rows.append({"exact_answer_probability": p, "method": method,
                         "question": q, "questions": int(q is not None),
                         "expected_loss": actual[q]["loss"] if q else risk(prior, losses)})
    # Independent closed-form checks for this two-bit construction.
    assert math.isclose(risk(prior, losses), 0.5)
    assert math.isclose(evaluate(prior, losses, channel("exact", 0.2))["loss"], 0.4)
    assert math.isclose(evaluate(prior, losses, channel("important", 0.9))["loss"], 0.095)
    assert math.isclose(evaluate(prior, losses, channel("minor", 1))["loss"], 0.45)
    # Controls: no information; already optimal action; non-uniform beliefs.
    for belief in [prior, [0.1, 0.2, 0.3, 0.4]]:
        assert math.isclose(evaluate(belief, losses, channel("exact", 0))["value"], 0)
        common_optimum = [[0.0] * 4, [1.0] * 4]
        for q in questions:
            assert math.isclose(evaluate(belief, common_optimum, channel(q, 1))["value"], 0)
            for p in [0, 0.2, 0.5, 1]:
                c = channel(q, p)
                assert all(math.isclose(sum(row.values()), 1) for row in c)
                assert math.isclose(sum(w for w, _ in posterior_branches(belief, c)), 1)
                value = evaluate(belief, losses, c)["value"]
                assert -1e-12 <= value <= risk(belief, losses) + 1e-12
    result = {
        "status": "constructed_diagnostic_not_empirical_evidence",
        "checks": "passed",
        "assumptions": ["exact prior", "Bayes-optimal executor shared by all selectors",
                        "true channel available to IG and EVSI selectors",
                        "one question at most", "zero question cost", "no learned method",
                        "actual channel used for every post-answer decision"],
        "rows": rows,
    }
    output = Path(__file__).resolve().parents[1] / "results" / "clarification_sanity.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("Analytic and probability checks passed.")
    print("Constructed case: exact-answer probability = 0.2")
    for row in rows:
        if row["exact_answer_probability"] == 0.2:
            print(f'{row["method"]}: question={row["question"]}, loss={row["expected_loss"]:.3f}')
    print(f"Results saved to {output}")


if __name__ == "__main__":
    run()
