from __future__ import annotations

import json
from pathlib import Path

from opengrad.results.registry import build_registry


def test_known_b0_external_layout_is_projected_without_fake_checkpoints(tmp_path: Path) -> None:
    run = tmp_path / "runs/tool_calling/qwen35_2b/baseline"
    run.mkdir(parents=True)
    reports = tmp_path / "reports/baselines/qwen35_2b_baseline"
    reports.mkdir(parents=True)
    for name in ("metrics.json", "predictions.jsonl", "residual-profile.json", "environment.json"):
        (reports / name).write_text("{}\n", encoding="utf-8")
    (run / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "tool_calling/qwen35_2b/baseline",
                "status": "EVALUATED",
                "training_algorithm": "evaluation",
                "model_id": "Qwen/Qwen3.5-2B",
                "model_revision": "rev",
                "tokenizer_revision": "rev",
                "metadata": {
                    "metrics": "reports/baselines/qwen35_2b_baseline/metrics.json",
                    "predictions": "reports/baselines/qwen35_2b_baseline/predictions.jsonl",
                    "residual_profile": "reports/baselines/qwen35_2b_baseline/residual-profile.json",
                    "environment": "reports/baselines/qwen35_2b_baseline/environment.json",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "runs/central_ledger.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs/central_ledger.jsonl").write_text("", encoding="utf-8")

    rows, findings = build_registry(tmp_path)

    assert not any(
        finding.code == "STATUS_WITHOUT_EVAL_ARTIFACTS" for finding in findings
    )
    row = rows[0]
    assert row.evaluated_checkpoints == []
    assert row.provenance["eval"] == "reports/baselines/qwen35_2b_baseline"
