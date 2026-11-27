"""Exact decision loss for the three-flight worked example."""
from fractions import Fraction as F
from itertools import product
from collections import Counter
import csv
from pathlib import Path

PRIOR = {"A": F(1, 2), "B": F(3, 10), "C": F(1, 5)}
UNKNOWN = {a: "unknown" for a in PRIOR}
EXACT = {a: a for a in PRIOR}
PARTIAL = {"A": "A", "B": "B/C", "C": "B/C"}
BINARY = {"A": "no", "B": "yes", "C": "no"}


def loss(channel):
    joint = {}
    for probability, answers in channel:
        for preference, mass in PRIOR.items():
            row = joint.setdefault(answers[preference], Counter())
            row[preference] += probability * mass
    return sum(sum(row.values()) - max(row.values()) for row in joint.values())


def main():
    immediate = loss([(1, UNKNOWN)])
    assert (immediate, loss([(1, EXACT)]), loss([(1, BINARY)])) == (F(1, 2), 0, F(1, 5))
    grid = [F(i, 4) for i in range(5)]
    rows = []
    for exact, partial, yes_no in product(grid, grid, grid[1:]):
        if exact + partial > 1:
            continue
        unknown = 1 - exact - partial
        q1 = loss([(exact, EXACT), (partial, PARTIAL), (unknown, UNKNOWN)])
        q2 = loss([(yes_no, BINARY), (1 - yes_no, UNKNOWN)])
        assert q1 == partial / 5 + unknown / 2
        assert q2 == F(1, 2) - yes_no * F(3, 10)
        winner = "Q1" if q1 < q2 else "Q2" if q2 < q1 else "tie"
        rows.append((exact, partial, unknown, yes_no, immediate, q1, q2,
                     winner, q1 - min(immediate, q1, q2)))
    assert len(rows) == 60
    output = Path(__file__).resolve().parents[1] / "results" / "flight_questions.csv"
    with output.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["q1_exact", "q1_partial", "q1_unknown", "q2_answer",
                         "no_question_loss", "q1_loss", "q2_loss", "best_question", "excess_loss"])
        writer.writerows(rows)
    print(Counter(row[-2] for row in rows))
    print("Mean and max excess loss:", sum(row[-1] for row in rows) / len(rows),
          max(row[-1] for row in rows))


if __name__ == "__main__":
    main()
