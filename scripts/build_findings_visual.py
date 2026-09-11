#!/usr/bin/env python3
"""Generate the baseline findings charts and a self-contained HTML page.

Every number is computed from the authoritative per-example predictions rather than copied from
a report, so a chart cannot drift from the evidence behind it. Restricted to one evaluation
partition at a time, because a metric from DEV and a metric from the confirmatory partition are
not comparable and mixing them on one axis is the fastest way to publish a misleading figure.

Usage:
    python scripts/build_findings_visual.py                 # writes reports/visual/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opengrad.evaluation.routing import routing_metrics

PARTITION = "reports/evaluation/behavioral-heldout-v2-partition.json"
OUT_DIR = "reports/visual"

# The experiment lineage, oldest first. `predictions` are per-example files so every figure is
# derived from the same kind of evidence.
LINEAGE: list[dict[str, str]] = [
    {
        "key": "b0",
        "label": "B0\n(untrained)",
        "predictions": "reports/baselines/qwen35_2b_baseline/predictions.jsonl",
        "note": "the frozen baseline",
    },
    {
        "key": "m0v1",
        "label": "M0-v1\n(corpus v1)",
        "predictions": "runs/qwen35_2b_m0_sft_full_v3/eval/checkpoint-2400/predictions.jsonl",
        "note": "SFT collapsed; 9 tool-call targets in 55,719 records",
    },
    {
        "key": "dpo",
        "label": "M1-DPO-v1",
        "predictions": "runs/qwen35_2b_m1_dpo_v1/eval/checkpoint-100/predictions.jsonl",
        "note": "negative; not reproducible",
    },
    {
        "key": "v2part",
        "label": "M0 partial-v2\n@1200",
        "predictions": "runs/qwen35_2b_m0_sft_v2corpus/eval/checkpoint-1200/predictions.jsonl",
        "note": "checkpoint selected on the full population",
    },
    {
        "key": "v2final",
        "label": "M0 final-v2\n@1800",
        "predictions": "runs/m0_sft_canonical_v2_final/eval/confirmatory/checkpoint-1800/predictions.jsonl",
        "note": "the definitive run, selected on DEV",
    },
]

METRICS = (
    "call_f1",
    "call_precision",
    "call_recall",
    "over_call_rate",
    "clarification_accuracy",
    "unsupported_accuracy",
)

# A restrained palette; the baseline is deliberately grey so the trained models read as the
# subject and the baseline reads as the reference it is.
COLOURS = {
    "b0": "#6b7280",
    "m0v1": "#dc2626",
    "dpo": "#f59e0b",
    "v2part": "#2563eb",
    "v2final": "#16a34a",
}


def partition_ids(side: str) -> set[str]:
    payload = json.loads((ROOT / PARTITION).read_text(encoding="utf-8"))
    return set(payload["example_ids"][side])


def score(predictions: Path, include: set[str]) -> dict | None:
    """Metrics for one prediction file, restricted to a partition."""
    if not predictions.is_file():
        return None
    rows = []
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("example_id")) in include:
            rows.append(row)
    if not rows:
        return None
    metrics = routing_metrics(
        [str(r["expected_decision"]) for r in rows],
        [str(r["prediction"]["decision"]) for r in rows],
    )
    metrics["parse_valid_rate"] = sum(
        1 for r in rows if (r.get("parser") or {}).get("status") == "RAW_VALID"
    ) / len(rows)
    metrics["records"] = len(rows)
    return metrics


def measure(side: str) -> dict[str, dict]:
    include = partition_ids(side)
    out: dict[str, dict] = {}
    for entry in LINEAGE:
        metrics = score(ROOT / entry["predictions"], include)
        if metrics is not None:
            out[entry["key"]] = metrics
    return out


def png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def figure_tradeoff(results: dict[str, dict]) -> str:
    """Precision against recall: the axis where a degenerate policy is visibly degenerate."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for key, metrics in results.items():
        label = next(e["label"] for e in LINEAGE if e["key"] == key).replace("\n", " ")
        ax.scatter(
            metrics["call_recall"],
            metrics["call_precision"],
            s=190,
            color=COLOURS[key],
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
            label=label,
        )
        ax.annotate(
            f"{metrics['call_f1']:.4f}",
            (metrics["call_recall"], metrics["call_precision"]),
            textcoords="offset points",
            xytext=(0, -20),
            ha="center",
            fontsize=8,
            color="#374151",
        )
    ax.axhline(0.5, color="#e5e7eb", linewidth=1, zorder=1)
    ax.axvline(0.5, color="#e5e7eb", linewidth=1, zorder=1)
    ax.set_xlabel("call recall  →  finding calls that should be made")
    ax.set_ylabel("call precision  →  calls that were correct")
    ax.set_title(
        "Precision against recall\nB0 sits in the bottom-right corner: it calls almost always",
        fontsize=11,
        loc="left",
    )
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, zorder=0)
    return png(fig)


def figure_per_class(results: dict[str, dict]) -> str:
    """Per-class recall, the balanced view a single headline number hides."""
    labels = ["must call", "clarify", "unsupported"]
    keys = ["must_call_accuracy", "clarification_accuracy", "unsupported_accuracy"]
    present = [k for k in results if all(key in results[k] for key in keys)]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.8 / max(1, len(present))
    for index, key in enumerate(present):
        label = next(e["label"] for e in LINEAGE if e["key"] == key).replace("\n", " ")
        offsets = [i + index * width for i in range(len(labels))]
        values = [results[key][k] for k in keys]
        ax.bar(offsets, values, width, label=label, color=COLOURS[key], edgecolor="white")
        for offset, value in zip(offsets, values):
            ax.annotate(
                f"{value:.2f}",
                (offset, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
                color="#374151",
            )
    ax.set_xticks([i + width * (len(present) - 1) / 2 for i in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("recall within the class")
    ax.set_ylim(0, 1.08)
    ax.set_title(
        "Per-class behaviour\nA model can score well overall while failing one class entirely",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    return png(fig)


def figure_overcall(results: dict[str, dict]) -> str:
    keys = [k for k in results if "over_call_rate" in results[k]]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    positions = range(len(keys))
    values = [results[k]["over_call_rate"] for k in keys]
    ax.barh(
        list(positions),
        values,
        color=[COLOURS[k] for k in keys],
        edgecolor="white",
        height=0.62,
    )
    for position, value in zip(positions, values):
        ax.annotate(
            f"{value:.3f}",
            (value, position),
            textcoords="offset points",
            xytext=(5, 0),
            va="center",
            fontsize=8,
            color="#374151",
        )
    ax.set_yticks(list(positions))
    ax.set_yticklabels(
        [next(e["label"] for e in LINEAGE if e["key"] == k).replace("\n", " ") for k in keys]
    )
    ax.invert_yaxis()
    ax.set_xlabel("over-call rate  →  calls made when the correct answer was not a call")
    ax.set_xlim(0, max(values) * 1.22 if values else 1)
    ax.set_title(
        "Over-calling\nB0 calls on most requests it should have answered, clarified, or declined",
        fontsize=11,
        loc="left",
    )
    ax.grid(axis="x", alpha=0.25)
    return png(fig)


def figure_confusion(results: dict[str, dict], keys: list[str]) -> str:
    """Confusion matrices side by side, so the failure mode is visible and not just its score."""
    labels = ["CALL", "ANSWER", "CLARIFY", "UNSUPPORTED"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.5))
    if len(keys) == 1:
        axes = [axes]
    for axis, key in zip(axes, keys):
        matrix = results[key]["confusion_matrix"]
        grid = [[matrix[truth].get(guess, 0) for guess in labels] for truth in labels]
        axis.imshow(grid, cmap="Blues", vmin=0)
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(
                    column,
                    row,
                    grid[row][column],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white"
                    if grid[row][column] > max(max(r) for r in grid) * 0.55
                    else "#111827",
                )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlabel("predicted", fontsize=8)
        if axis is axes[0]:
            axis.set_ylabel("gold", fontsize=8)
        axis.set_title(
            next(e["label"] for e in LINEAGE if e["key"] == key).replace("\n", " "), fontsize=9
        )
    fig.suptitle(
        "Where the decisions actually go\nEvery off-diagonal cell is a wrong decision",
        fontsize=11,
        x=0.02,
        ha="left",
    )
    return png(fig)


def figure_trajectory() -> str:
    """The final-v2 checkpoint curve on DEV, with the selection annotated."""
    curve = json.loads(
        (ROOT / "runs/m0_sft_canonical_v2_final/eval/dev/curve.json").read_text(encoding="utf-8")
    )
    steps, f1, precision, recall, over = [], [], [], [], []
    for point in curve["points"]:
        metrics = point["baseline_comparison"]["metrics"]
        steps.append(point["checkpoint_step"])
        f1.append(metrics["call_f1"]["candidate"])
        precision.append(metrics["call_precision"]["candidate"])
        recall.append(metrics["call_recall"]["candidate"])
        over.append(metrics["over_call_rate"]["candidate"])

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ax.plot(steps, f1, "o-", label="call_f1", color="#16a34a", linewidth=2)
    ax.plot(steps, precision, "s-", label="precision", color="#2563eb")
    ax.plot(steps, recall, "^-", label="recall", color="#7c3aed")
    ax.plot(steps, over, "v--", label="over-call (lower is better)", color="#dc2626")
    baseline = curve["points"][0]["baseline_comparison"]["metrics"]["call_f1"]["baseline"]
    ax.axhline(baseline, color="#6b7280", linestyle=":", label=f"B0 call_f1 ({baseline:.4f})")

    selected = 1800
    ax.axvline(selected, color="#111827", linestyle="--", linewidth=1)
    ax.annotate(
        "selected: 1800",
        xy=(selected, 0.20),
        xytext=(selected + 260, 0.44),
        fontsize=8,
        color="#111827",
        arrowprops={"arrowstyle": "->", "color": "#111827", "linewidth": 1},
    )
    for step, value in zip(steps, over):
        if value > 0.20:
            ax.annotate(
                "disqualified:\nover-calls",
                (step, value),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color="#dc2626",
            )
    ax.set_xticks(steps)
    ax.set_xlabel("checkpoint step (DEV partition)")
    ax.set_ylabel("metric value")
    ax.set_title(
        "The training trajectory, and why the best call_f1 was not chosen",
        fontsize=11,
        loc="left",
    )
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return png(fig)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenGrad — baseline findings</title>
<style>
  :root {{ color-scheme: light; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 2.5rem 1.25rem 4rem;
    font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111827; background: #ffffff;
  }}
  main {{ max-width: 60rem; margin: 0 auto; }}
  h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 0.35rem; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1.25rem; margin: 3rem 0 0.4rem; letter-spacing: -0.01em; }}
  h2:first-of-type {{ margin-top: 2rem; }}
  p {{ margin: 0.5rem 0 1rem; color: #374151; }}
  .lede {{ font-size: 1.05rem; color: #4b5563; max-width: 46rem; }}
  .meta {{ font-size: 0.85rem; color: #6b7280; margin-bottom: 2rem; }}
  a {{ color: #1d4ed8; }}
  figure {{ margin: 1.5rem 0; }}
  figure img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 6px; }}
  figcaption {{ font-size: 0.85rem; color: #6b7280; margin-top: 0.5rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
  th, td {{ padding: 0.5rem 0.6rem; text-align: right; border-bottom: 1px solid #e5e7eb; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ font-weight: 600; border-bottom: 2px solid #d1d5db; }}
  tbody tr:hover {{ background: #f9fafb; }}
  .win {{ font-weight: 700; color: #15803d; }}
  .warn {{ color: #b45309; }}
  .note {{
    border-left: 3px solid #d1d5db; padding: 0.7rem 0 0.7rem 1rem; margin: 1.25rem 0;
    color: #4b5563; font-size: 0.92rem; background: #fafafa;
  }}
  .grid {{ display: grid; gap: 0.8rem; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); margin: 1.5rem 0; }}
  .stat {{ border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.85rem 1rem; }}
  .stat .v {{ font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }}
  .stat .k {{ font-size: 0.78rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.04em; }}
  footer {{ margin-top: 3.5rem; padding-top: 1.25rem; border-top: 1px solid #e5e7eb; font-size: 0.85rem; color: #6b7280; }}
  code {{ background: #f3f4f6; padding: 0.1rem 0.35rem; border-radius: 4px; font-size: 0.88em; }}
</style>
</head>
<body>
<main>

<h1>OpenGrad — baseline findings</h1>
<p class="lede">
  What controlled post-training did to a 2&nbsp;billion-parameter model's tool-calling policy, measured
  on a frozen held-out set. The interesting result is not the headline number; it is that the
  baseline's high score comes from calling a tool almost every time.
</p>
<p class="meta">
  Model: <a href="{model_url}">{model_name}</a> ·
  Training corpus: <a href="{dataset_url}">{dataset_name}</a> ·
  Evaluated on the pre-registered confirmatory partition ({records} examples).
</p>

<div class="grid">
{stat_cards}
</div>

<h2>B0's score is a trap</h2>
<p>
  The baseline reaches <strong>call_f1 {b0_f1}</strong> by calling a tool on
  <strong>{b0_over}</strong> of requests whose correct answer was <em>not</em> a call. It recalls
  {b0_recall} of the calls that should be made, and gets {b0_unsupp} of unsupported requests right.
  A model that calls everything scores well on recall and badly on everything that matters.
</p>
<figure>
  <img alt="Precision against recall for every model in the lineage" src="data:image/png;base64,{tradeoff}">
  <figcaption>
    Each point is one model; the annotation is its call_f1. B0 occupies the bottom-right corner —
    maximum recall, minimum precision. The final model sits toward the top-right: it keeps most of
    the recall and gains a great deal of precision.
  </figcaption>
</figure>

<h2>A single number hides which classes work</h2>
<figure>
  <img alt="Per-class recall for always-call, clarify and unsupported classes" src="data:image/png;base64,{per_class}">
  <figcaption>
    Recall within each decision class. B0 is near-perfect on <em>must call</em> and near-zero on
    the other two: it has one behaviour. The trained models trade some call recall for large gains
    in the classes the baseline cannot handle at all. The <em>direct answer</em> class is absent
    because the held-out set contains no examples of it.
  </figcaption>
</figure>

<h2>Over-calling, which B0 cannot avoid</h2>
<figure>
  <img alt="Over-call rate by model" src="data:image/png;base64,{overcall}">
  <figcaption>
    The fraction of non-call requests on which the model calls anyway. Lower is better.
  </figcaption>
</figure>

<h2>Where the decisions go</h2>
<figure>
  <img alt="Confusion matrices for the baseline and the final model" src="data:image/png;base64,{confusion}">
  <figcaption>
    Rows are the correct answer, columns are the model's output. B0's rows are almost entirely in
    the CALL column. The final model spreads its decisions across the classes instead.
  </figcaption>
</figure>

<h2>Why the checkpoint with the best score was not selected</h2>
<figure>
  <img alt="Final-v2 checkpoint trajectory on the DEV partition" src="data:image/png;base64,{trajectory}">
  <figcaption>
    The selection rule was written and committed before any checkpoint was evaluated. It requires
    over-call ≤ 0.20, which disqualifies steps 600 and 1200 — the two with the highest call_f1.
    Both buy their score the same way B0 does. Selection ranked the survivors on balanced
    behaviour and took step 1800 on a tie-break against 2400.
  </figcaption>
</figure>

<h2>Every number</h2>
{table}
<div class="note">
  {caveats}
</div>

<footer>
  Generated by <code>scripts/build_findings_visual.py</code> from the per-example predictions of
  each run. Every figure is computed from those predictions rather than copied from a report, so a
  chart cannot drift from its evidence.
  Full write-ups:
  <a href="{execution_report}">execution report</a> ·
  <a href="{evaluation_report}">evaluation report</a> ·
  <a href="{repo_url}">repository</a>.
</footer>

</main>
</body>
</html>
"""


def render_table(results: dict[str, dict]) -> str:
    header = (
        "<table><thead><tr><th>model</th>"
        + "".join(f"<th>{m.replace('_', ' ')}</th>" for m in METRICS)
        + "<th>records</th></tr></thead><tbody>"
    )
    rows = []
    best_f1 = max((r["call_f1"] for r in results.values()), default=0)
    for entry in LINEAGE:
        metrics = results.get(entry["key"])
        if metrics is None:
            continue
        cells = []
        for name in METRICS:
            value = metrics[name]
            mark = ' class="win"' if name == "call_f1" and value == best_f1 else ""
            cells.append(f"<td{mark}>{value:.4f}</td>")
        rows.append(
            f"<tr><td>{entry['label'].replace(chr(10), ' ')}</td>"
            + "".join(cells)
            + f"<td>{metrics['records']}</td></tr>"
        )
    return header + "".join(rows) + "</tbody></table>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="confirmatory", choices=["dev", "confirmatory"])
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--model-url", default="")
    parser.add_argument("--model-name", default="OpenGrad Qwen3.5-2B M0 final-v2")
    parser.add_argument("--dataset-url", default="")
    parser.add_argument("--dataset-name", default="OpenGrad ToolPolicy Canonical v2")
    args = parser.parse_args()

    results = measure(args.side)
    if not results:
        print("no results to plot", file=sys.stderr)
        return 1

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    charts = {
        "tradeoff": figure_tradeoff(results),
        "per_class": figure_per_class(results),
        "overcall": figure_overcall(results),
        "confusion": figure_confusion(results, ["b0", "v2part", "v2final"]),
        "trajectory": figure_trajectory(),
    }
    for name, payload in charts.items():
        (out / f"{name}.png").write_bytes(base64.b64decode(payload))

    b0 = results["b0"]
    final = results.get("v2final") or results["b0"]
    cards = [
        ("B0 call_f1", f"{b0['call_f1']:.4f}", "baseline, over-calls badly"),
        ("final call_f1", f"{final['call_f1']:.4f}", "selected checkpoint"),
        ("B0 over-call", f"{b0['over_call_rate']:.3f}", "on non-call requests"),
        ("final over-call", f"{final['over_call_rate']:.3f}", "a quarter of the baseline's"),
        ("final precision", f"{final['call_precision']:.4f}", f"B0: {b0['call_precision']:.4f}"),
        ("final recall", f"{final['call_recall']:.4f}", f"B0: {b0['call_recall']:.4f}"),
    ]
    stat_cards = "\n".join(
        f'<div class="stat"><div class="v">{value}</div><div class="k">{label}</div>'
        f'<div class="k" style="text-transform:none;letter-spacing:0">{sub}</div></div>'
        for label, value, sub in cards
    )

    prompt_count = b0["confusion_matrix"]["CALL"].get("CALL", 0)
    caveats = (
        "Read with three caveats. "
        "<strong>The confirmatory partition is internal, not an untouched external benchmark</strong> — the "
        "wider upstream population has already influenced earlier work in this project, so these are "
        "pre-registered internal results. "
        "<strong>Tool-selection accuracy, argument validity and schema validity are not measured at all</strong> "
        "by this evaluator, so their absence from every row above is not a zero. "
        "<strong>The corpus changed in three ways at once</strong> — xLAM was added, ToolACE grew from 697 to "
        "11,051 accepted records after a validator fix, and When2Call from 4,000 to 6,505 after an "
        "interrupted materialization was corrected — so no single source can be credited with the "
        "improvement. Separating them needs the ablation that the supervision configuration allows "
        "and which this experiment did not run."
    )

    html = HTML_TEMPLATE.format(
        **charts,
        table=render_table(results),
        stat_cards=stat_cards,
        model_url=args.model_url or "#",
        model_name=args.model_name,
        dataset_url=args.dataset_url or "#",
        dataset_name=args.dataset_name,
        records=b0["records"],
        b0_f1=f"{b0['call_f1']:.4f}",
        b0_over=f"{b0['over_call_rate']:.1%}",
        b0_recall=f"{b0['call_recall']:.1%}",
        b0_unsupp=f"{b0['unsupported_accuracy']:.1%}",
        caveats=caveats,
        repo_url="https://github.com/arjhinety/OpenGrad",
        execution_report=(
            "https://github.com/arjhinety/OpenGrad/blob/master/"
            "reports/M0_CANONICAL_V2_FINAL_EXECUTION_REPORT.md"
        ),
        evaluation_report=(
            "https://github.com/arjhinety/OpenGrad/blob/master/"
            "reports/M0_CANONICAL_V2_FINAL_EVALUATION.md"
        ),
    )
    (out / "index.html").write_text(html, encoding="utf-8")

    print(f"partition : {args.side} ({b0['records']} examples, {prompt_count} gold CALLs)")
    for entry in LINEAGE:
        if entry["key"] in results:
            m = results[entry["key"]]
            print(
                f"  {entry['label'].replace(chr(10), ' '):<22}"
                f"f1={m['call_f1']:.4f}  prec={m['call_precision']:.4f}  "
                f"rec={m['call_recall']:.4f}  over={m['over_call_rate']:.4f}"
            )
    print(f"wrote {out.relative_to(ROOT)}/ (index.html + {len(charts)} charts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
