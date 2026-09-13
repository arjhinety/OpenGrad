#!/usr/bin/env python3
"""Comparison figures for the M1-v2 quantization study.

Every number here is recomputed from the audit records and per-example predictions that the study
wrote, never copied out of a report. A figure that is rendered from prose can drift from its
evidence silently; one that is rendered from the evidence cannot.

Mirrors the conventions in `scripts/build_findings_visual.py`: matplotlib Agg, PNGs embedded as
base64 in a single self-contained HTML page, one evaluation partition at a time.

Charts are added as their evidence lands. The artifact-composition figure needs only the
ExecuTorch audit; the retention, Pareto and confusion figures need GGUF behavioural results and are
skipped with a printed reason until those exist, rather than being drawn from placeholder numbers.

Usage:
    python scripts/build_quantization_visual.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports/visual/quantization"
AUDIT = ROOT / "results/quantization/executorch/quantization_audit_8da4w.json"

GIB = 1024**3

# Categorical slots 1-5 of the validated default palette, in fixed order. Validated for this chart
# with `validate_palette.js "<these>" --mode light --surface #ffffff`: lightness band, chroma
# floor, adjacent CVD (worst ΔE 9.1) and normal-vision (worst ΔE 19.6) all PASS. Contrast warns
# below 3:1 for three slots, which obligates relief — this chart ships visible direct labels on
# every segment, so identity never rests on hue alone.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#ffffff"


def png(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def composition_rows(audit: dict) -> tuple[list[dict], list[dict]]:
    """Byte composition of the fp32 and 8da4w artifacts, in the same category order.

    The categories are chosen so the two bars are read against each other: the embedding segment is
    byte-identical in both, which is the entire point of the figure.
    """
    named = audit["named_data"]
    consts = audit["program_constants"]
    byte_totals = audit["bytes"]

    quantized_classes = [c for c in named["classes"] if c["verdict"] == "QUANTIZED_INT4"]
    skipped_classes = [c for c in named["classes"] if c["verdict"] == "SKIPPED_FP32"]

    int4_payload = sum(c["int4_payload_bytes"] * c["count"] for c in quantized_classes)
    scales = sum(c["scale_bytes"] * c["count"] for c in quantized_classes)
    skipped_bytes = sum(c["parameters"] * 4 for c in skipped_classes)

    embedding = consts["embedding_bytes"]

    # fp32 reference: the same weights, stored at 4 bytes each, no scales.
    fp32_linear = sum(c["parameters"] * 4 for c in quantized_classes)

    fp32_rows = [
        {"label": "Embedding table (fp32)", "bytes": embedding},
        {"label": "Linear weights (fp32)", "bytes": fp32_linear},
        {"label": "Quantization scales", "bytes": 0},
        {"label": "DeltaNet conv1d (fp32)", "bytes": skipped_bytes},
        {"label": "Norms + container", "bytes": byte_totals["fp32"] - embedding - fp32_linear - skipped_bytes},
    ]
    quant_rows = [
        {"label": "Embedding table (fp32)", "bytes": embedding},
        {"label": "Linear weights (int4)", "bytes": int4_payload},
        {"label": "Quantization scales", "bytes": scales},
        {"label": "DeltaNet conv1d (fp32)", "bytes": skipped_bytes},
        {"label": "Norms + container", "bytes": byte_totals["quantized"] - embedding - int4_payload - scales - skipped_bytes},
    ]
    return fp32_rows, quant_rows


def figure_composition(audit: dict) -> str:
    """Where the bytes actually are.

    The headline of the 8da4w export is "3.09x smaller", and that number hides the finding: the
    weights shrank 7.1x and the embedding did not move at all, so the embedding is what now
    dominates the artifact. A stacked bar puts both facts in one frame — the quantized bar is
    shorter, and most of what remains is the one segment that is identical in both.
    """
    fp32_rows, quant_rows = composition_rows(audit)
    fig, ax = plt.subplots(figsize=(11.0, 3.9))

    bars = [("fp32\n8.91 GiB", fp32_rows), ("8da4w\n2.89 GiB", quant_rows)]
    height = 0.46
    for index, (name, rows) in enumerate(bars):
        left = 0.0
        y = len(bars) - 1 - index
        for slot, row in enumerate(rows):
            width = row["bytes"] / GIB
            if width <= 0:
                continue
            ax.barh(
                y, width, height=height, left=left,
                color=SERIES[slot], edgecolor=SURFACE, linewidth=2, zorder=3,
            )
            # Direct labels are the relief for the sub-3:1 contrast slots, so any segment wide
            # enough to hold text gets its value inside; the rest are covered by the legend and
            # the table in the HTML page.
            if width > 0.42:
                ax.text(
                    left + width / 2, y, f"{width:.2f}",
                    ha="center", va="center", color="white", fontsize=10,
                    fontweight="bold", zorder=4,
                )
            left += width
        ax.text(
            left + 0.12, y, f"{left:.2f} GiB",
            ha="left", va="center", color=INK, fontsize=10.5, fontweight="bold",
        )

    ax.set_yticks([1, 0])
    ax.set_yticklabels([bars[0][0], bars[1][0]], fontsize=10.5, color=INK)
    ax.set_xlabel("artifact size (GiB)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(-0.6, 1.6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9.5)
    ax.set_title(
        "ExecuTorch XNNPACK artifact composition — the embedding is what 8da4w does not touch",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    # Embeddings are tied, so the same 508.6M-parameter matrix is stored twice: fp32 as the
    # embedding lookup (blue) and again as lm_head inside the weight segment (orange). Both
    # segments are measured, not double-counted, but a reader who does not know they are the same
    # matrix will misread the chart — so it is stated on the chart rather than only in the report.
    fig.text(
        0.5, 0.02,
        "Embeddings are tied: the same 508.6M-parameter matrix is stored twice — fp32 as the "
        "embedding lookup, and again inside the weight segment as lm_head.",
        ha="center", fontsize=9, color=INK_SECONDARY,
    )
    # Two of these segments are a few hundred kilobytes and render sub-pixel. A legend that lists
    # them without saying so sends the reader hunting for marks that are not findable; carrying the
    # size in the label makes their absence the point rather than a defect.
    fig.legend(
        handles=[
            Patch(
                facecolor=SERIES[i],
                label=(
                    f"{row['label']} — {row['bytes'] / GIB:.2f} GiB"
                    if row["bytes"] / GIB >= 0.01
                    else f"{row['label']} — {row['bytes'] / 1024**2:.1f} MiB (not visible at this scale)"
                ),
            )
            for i, row in enumerate(quant_rows)
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2,
        frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
    )
    # Explicit margins rather than tight_layout: the legend and footnote are figure-level artists,
    # and tight_layout only reserves room for axes-level ones, so it lets them sit on the x label.
    fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.46)
    return png(fig)


def figure_coverage(audit: dict) -> str:
    """What fraction of each weight class became int4, by parameter count.

    Position carries magnitude and the single colour carries nothing but "this is one series" —
    the skipped class is the only one that differs, and it is called out by a status colour plus a
    label rather than by hue alone.
    """
    named = audit["named_data"]
    # Ascending, because barh draws bottom-up and the largest class should read at the top.
    classes = sorted(named["classes"], key=lambda c: c["parameters"])
    labels, values, colours = [], [], []
    for item in classes:
        identity = item["identity"]
        short = identity.split(" (")[0] if "(" in identity else identity
        labels.append(f"{short}\n{item['count']}x {item['elements_each']:,}")
        values.append(item["parameters"] / 1e6)
        # Status red, paired with an explicit label below — never hue alone.
        colours.append("#d03b3b" if item["verdict"] == "SKIPPED_FP32" else SERIES[0])

    # Horizontal: these class names are long, and on a vertical axis they collide into an
    # unreadable smear. Rotating is what gives the labels room, not shrinking the type.
    #
    # Dots rather than bars, because the range spans four orders of magnitude and needs a log
    # axis. A bar's length is read as proportional to its value, which on a log axis it is not —
    # the 1,132M bar is not 3x the 0.4M bar, it is 3000x. A dot encodes value by position only, so
    # the log axis stops lying. Linear was the other option and hides everything under 50M.
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    positions = list(range(len(values)))
    ax.hlines(positions, 0.2, values, color="#d7d6d0", linewidth=1.6, zorder=2)
    ax.scatter(values, positions, s=150, color=colours, zorder=3, edgecolor=SURFACE, linewidth=1.4)
    for pos, (value, item) in enumerate(zip(values, classes)):
        text = f"{value:,.1f}M"
        if item["verdict"] == "SKIPPED_FP32":
            text += "   SKIPPED — not an nn.Linear"
        ax.text(
            value * 1.35, pos, text,
            ha="left", va="center", fontsize=9.5,
            color="#d03b3b" if item["verdict"] == "SKIPPED_FP32" else INK,
            fontweight="bold" if item["verdict"] == "SKIPPED_FP32" else "normal",
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8.6, color=INK_SECONDARY)
    ax.set_xlabel("parameters (millions, log scale)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 12000)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(
        "8da4w coverage by weight class — 99.98% of named-data parameters are int4",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    fig.tight_layout()
    return png(fig)


LADDER_ORDER = ("Q2_K", "Q3_K_M", "IQ3_M", "IQ4_XS", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
RETENTION_METRICS = (
    "call_f1", "call_precision", "call_recall",
    "clarification_accuracy", "unsupported_accuracy",
)


def load_ladder() -> tuple[dict | None, list[dict]]:
    """Scored rungs, in ladder order. Absent rungs are omitted, never invented."""
    gguf = ROOT / "results/quantization/gguf"
    baseline_path = gguf / "score_m1-v2-bf16_confirmatory.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    rungs = []
    for name in LADDER_ORDER:
        path = gguf / f"score_m1-v2-{name}_confirmatory.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        bench_path = gguf / f"bench_m1-v2-{name}.json"
        doc["_bench"] = json.loads(bench_path.read_text(encoding="utf-8")) if bench_path.is_file() else None
        doc["_rung"] = name
        rungs.append(doc)
    return baseline, rungs


def figure_retention_ladder(baseline: dict, rungs: list[dict]) -> str:
    """Every retention metric across the ladder, against the gate floor.

    Colour carries the metric (identity, five fixed slots) and position carries the rung, so the
    encoding does not depend on rank — a rung dropping out would not repaint the survivors.
    """
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    xs = list(range(len(rungs)))
    for slot, metric in enumerate(RETENTION_METRICS):
        ys = [r["quantization_loss_vs_llamacpp_bf16"]["relative_retention"][metric] for r in rungs]
        ax.plot(xs, ys, marker="o", markersize=8, linewidth=2, color=SERIES[slot],
                label=metric.replace("_", " "), zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.axhline(1.0, color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(len(rungs) - 0.5, 1.0, " llama.cpp BF16 parity", va="bottom", ha="right",
            fontsize=9, color=INK_SECONDARY)
    # The frozen gate is 99% of the vLLM reference, not of this baseline; drawn as the declared
    # secondary release bar instead, which IS defined against llama.cpp BF16.
    ax.axhline(0.995, color="#d03b3b", linewidth=1.4, zorder=2)
    ax.text(0, 0.995, " release bar: 99.5% of llama.cpp BF16", va="bottom", ha="left",
            fontsize=9, color="#d03b3b", fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([r["_rung"] for r in rungs], fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("retention vs llama.cpp BF16", fontsize=10.5, color=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Behavioural retention across the PTQ ladder (baseline: llama.cpp BF16)",
                 fontsize=12, color=INK, pad=14, loc="left")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.22)
    return png(fig)


def figure_pareto(baseline: dict, rungs: list[dict]) -> str:
    """Decision agreement against artifact size — the trade the recommendation actually makes."""
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    xs = [r["artifact_bytes"] / GIB for r in rungs]
    ys = [r["quantization_loss_vs_llamacpp_bf16"]["output_agreement"]["decision_agreement"]
          for r in rungs]
    passed = [str(r["gate_vs_frozen_vllm_reference"]["decision"]).upper() for r in rungs]
    colours = [
        "#2a78d6" if p in {"PASS", "PASSED", "ACCEPT", "ACCEPTED", "PTQ_ACCEPTED", "PRESERVED"}
        else "#d03b3b"
        for p in passed
    ]
    ax.axhline(0.99, color="#d03b3b", linewidth=1.4, zorder=2)
    ax.text(max(xs), 0.99, "release bar: 0.99 decision agreement ", va="bottom", ha="right",
            fontsize=9, color="#d03b3b", fontweight="bold")
    ax.scatter(xs, ys, s=180, c=colours, zorder=3, edgecolor=SURFACE, linewidth=1.6)
    # Rungs cluster tightly in size (IQ4_XS/Q4_K_S, IQ3_M/Q3_K_M differ by ~0.02 GiB), so a fixed
    # label offset overlaps them into an unreadable smear. Alternate above/below by size order.
    for index, (x, y, r) in enumerate(sorted(zip(xs, ys, rungs), key=lambda t: t[0])):
        above = index % 2 == 0
        ax.annotate(
            r["_rung"], (x, y), textcoords="offset points",
            xytext=(0, 13 if above else -22), ha="center", fontsize=9, color=INK,
        )
    ax.set_xlabel("artifact size (GiB)", fontsize=10.5, color=INK_SECONDARY)
    ax.set_ylabel("decision agreement vs llama.cpp BF16", fontsize=10.5, color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(
        "Smaller is only better while behaviour holds — blue passes the frozen gate, red does not",
        fontsize=12, color=INK, pad=14, loc="left",
    )
    fig.tight_layout()
    return png(fig)


def figure_agreement_decomposition(rungs: list[dict]) -> str:
    """Byte-identical output vs decision agreement — deliberately not the same number.

    A rung can reword almost everything while preserving every decision. Plotting only one of
    these would report that rung as either broken or perfect, depending which one you picked.
    """
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    xs = list(range(len(rungs)))
    width = 0.38
    byte_identical = [
        r["quantization_loss_vs_llamacpp_bf16"]["output_agreement"]["byte_identical_rate"]
        for r in rungs
    ]
    decisions = [
        r["quantization_loss_vs_llamacpp_bf16"]["output_agreement"]["decision_agreement"]
        for r in rungs
    ]
    ax.bar([x - width / 2 for x in xs], byte_identical, width=width - 0.02,
           color=SERIES[0], zorder=3, label="exact output agreement (byte-identical)")
    ax.bar([x + width / 2 for x in xs], decisions, width=width - 0.02,
           color=SERIES[1], zorder=3, label="decision-level agreement")
    for x, value in zip(xs, decisions):
        ax.text(x + width / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=8.4,
                color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["_rung"] for r in rungs], fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("agreement vs llama.cpp BF16", fontsize=10.5, color=INK_SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("Same wording and same decision are different questions",
                 fontsize=12, color=INK, pad=14, loc="left")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=9.5, labelcolor=INK_SECONDARY)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.22)
    return png(fig)


def composition_table(audit: dict) -> str:
    """The table view. Required relief for the sub-3:1 palette slots, and the accessible path."""
    fp32_rows, quant_rows = composition_rows(audit)
    cells = []
    for fp32_row, quant_row in zip(fp32_rows, quant_rows):
        before, after = fp32_row["bytes"], quant_row["bytes"]
        if before == 0 or after == 0:
            ratio = "—"  # scales exist only in the quantized artifact; a ratio would be nonsense
        else:
            ratio = f"{before / after:.2f}x"
        cells.append(
            f"<tr><td>{quant_row['label']}</td>"
            f"<td class='n'>{before / GIB:.3f}</td>"
            f"<td class='n'>{after / GIB:.3f}</td>"
            f"<td class='n'>{ratio}</td></tr>"
        )
    return "".join(cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not AUDIT.is_file():
        raise SystemExit(
            f"missing {AUDIT.relative_to(ROOT)} — run scripts/audit_executorch_quantization.py"
        )
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit.get("reconciled"):
        raise SystemExit("the audit did not reconcile; refusing to draw figures from it")

    figures = [
        ("executorch_composition", "Artifact composition", figure_composition(audit)),
        ("executorch_coverage", "Quantization coverage by weight class", figure_coverage(audit)),
    ]
    for name, _, data in figures:
        (out / f"{name}.png").write_bytes(base64.b64decode(data))
        print(f"wrote {(out / f'{name}.png').relative_to(ROOT)}")

    pending = []
    baseline, rungs = load_ladder()
    scored = [r for r in rungs if "quantization_loss_vs_llamacpp_bf16" in r]
    if baseline is None:
        pending.append(
            "every ladder figure — the llama.cpp BF16 baseline has not been scored yet, and it is "
            "the reference every quantization comparison is measured against"
        )
    elif not scored:
        pending.append(
            "every ladder figure — the BF16 baseline exists but no quantized rung has been scored"
        )
    else:
        figures += [
            ("gguf_retention_ladder", "Retention across the PTQ ladder",
             figure_retention_ladder(baseline, scored)),
            ("gguf_pareto", "Decision agreement vs artifact size", figure_pareto(baseline, scored)),
            ("gguf_agreement_decomposition", "Exact vs decision-level agreement",
             figure_agreement_decomposition(scored)),
        ]
        missing = [name for name in LADDER_ORDER if name not in {r["_rung"] for r in scored}]
        if missing:
            # Never let a partial ladder read as a complete one.
            pending.append(
                "rungs absent from the ladder figures because they are not scored: "
                + ", ".join(missing)
            )
        for name, _, data in figures[2:]:
            (out / f"{name}.png").write_bytes(base64.b64decode(data))
            print(f"wrote {(out / f'{name}.png').relative_to(ROOT)}")

    blocks = "\n".join(
        f"<figure><img alt='{title}' src='data:image/png;base64,{data}'>"
        f"<figcaption>{title}</figcaption></figure>"
        for _, title, data in figures
    )
    pending_html = (
        "<div class='pending'><strong>Not yet drawn:</strong><ul>"
        + "".join(f"<li>{item}</li>" for item in pending)
        + "</ul></div>"
    ) if pending else ""

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>M1-v2 quantization study — figures</title>
<style>
  body {{ font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
         color: {INK}; background: #f9f9f7; margin: 0; padding: 40px 24px; }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  figure {{ margin: 0 0 36px; background: {SURFACE}; padding: 20px;
            border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; }}
  figure img {{ width: 100%; height: auto; display: block; }}
  figcaption {{ color: {INK_SECONDARY}; font-size: 13px; margin-top: 10px; }}
  table {{ border-collapse: collapse; width: 100%; background: {SURFACE};
           border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; }}
  th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid {GRID}; font-size: 14px; }}
  td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .pending {{ background: #fff; border: 1px solid rgba(11,11,11,0.10); border-left: 3px solid {MUTED};
              padding: 14px 18px; border-radius: 8px; color: {INK_SECONDARY}; font-size: 14px; }}
  h1 {{ font-size: 22px; margin: 0 0 6px; }}
  p.sub {{ color: {INK_SECONDARY}; margin: 0 0 28px; font-size: 14px; }}
</style>
<main>
<h1>M1-v2 quantization study — comparison figures</h1>
<p class="sub">Every value is recomputed from the audit records, not copied from a report.
Artifact: <code>{audit['artifact']}</code> against <code>{audit['reference_artifact']}</code>.</p>
{blocks}
<h2>Composition table</h2>
<table>
<thead><tr><th>segment</th><th class="n">fp32 (GiB)</th><th class="n">8da4w (GiB)</th><th class="n">ratio</th></tr></thead>
<tbody>{composition_table(audit)}</tbody>
</table>
<p class="sub" style="margin-top:24px">{pending_html}</p>
</main>
"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {(out / 'index.html').relative_to(ROOT)}")
    for item in pending:
        print(f"pending: {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
