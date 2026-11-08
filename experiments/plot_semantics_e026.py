"""Plot the declared illustrative E025-04 diagnostic from saved E026 results."""
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/userbench_semantics_e026"


def main():
    rows = json.loads((OUT / "mask_probes.json").read_text())
    case = "E025-04"
    modes = ["full", "first_only", "second_only"]
    questions = ["flight", "lodging", "rental_car", "restaurant"]
    labels = ["Flight", "Accommodation", "Rental car", "Restaurant"]
    titles = ["Both requirements revealed", "Only first requirement", "Only second requirement"]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "E026-fixed", "font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
    for ax, mode, title in zip(axes, modes, titles):
        values = []
        for q in questions:
            row = next(r for r in rows if r["case_id"] == case and r["question_id"] == q and r["mask"] == mode)
            assert row["delta_bounds"]["lower"] == row["delta_bounds"]["upper"]
            values.append(100 * float(Fraction(row["delta_bounds"]["lower"])))
        best = max(values)
        ax.barh(range(4), values, height=0.58, color=["#137c66" if v == best else "#bcc7cf" for v in values])
        for i, value in enumerate(values):
            ax.text(value + 0.45, i, "%.2f" % value, va="center", fontsize=9)
        ax.set_title(title, fontsize=11, pad=15)
        ax.set_yticks(range(4), labels)
        ax.set_xlim(0, 28)
        ax.set_xticks([0, 10, 20])
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    fig.suptitle("Which question helps most depends on what the reply reveals", fontsize=14, x=0.52, y=0.98)
    fig.supxlabel("Expected gain in mean criterion satisfaction across four aspects (percentage points)", fontsize=10, y=0.13)
    fig.text(0.5, 0.035, "E025-04 only • symbolic filtering and uniform choice • provisional criteria • not learned-agent performance", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.79, bottom=0.23, wspace=0.13)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    svg = buffer.getvalue()
    dest = OUT / "question_reversal.svg"
    if dest.exists():
        assert dest.read_bytes() == svg
    else:
        dest.write_bytes(svg)
    cache = Path.home() / ".cache/clarification-research/e026-question-reversal.png"
    fig.savefig(cache, dpi=160)
    manifest = {"matplotlib_version": matplotlib.__version__,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256((OUT / "mask_probes.json").read_bytes()).hexdigest(),
                "svg_sha256": hashlib.sha256(svg).hexdigest(),
                "selection": "illustrative first robust reversal after outcomes; not a preselected confirmatory case"}
    content = json.dumps(manifest, indent=2) + "\n"
    target = OUT / "figure_manifest.json"
    if target.exists():
        assert target.read_text() == content
    else:
        target.write_text(content)
    print(json.dumps({"svg": str(dest), "preview": str(cache), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
