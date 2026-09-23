"""Consistency checks between the documentation and the authoritative state.

The README, the roadmap, the reports and ``results/registry.jsonl`` are edited independently and can
drift apart. These tests pin the invariants that make such drift fail loudly:

* a mock or scaffold record can never be read as a real trained or promoted model;
* every valid promotion points at a checkpoint the checkpoint registry also calls promoted;
* the promoted DPO result is published, and its quote matches the release record;
* the documented dataset figures and the benchmark counting convention match the registries;
* the generated status document is byte-identical to a fresh regeneration;
* the stale claims this audit removed cannot silently return.

Failure messages name the authoritative artifact that disagrees, because the point of the test is
to say which edit is stale rather than only that something changed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))

from generate_experiment_status import build_document

from opengrad.results.registry import validate_registry

REGISTRY = ROOT / "results" / "registry.jsonl"
CHECKPOINT_REGISTRY = ROOT / "runs" / "checkpoint_registry.json"

# The identifier set the README's counting convention calls non-tiered. Adding or removing a
# benchmark here is a documentation change: update README's benchmark inventory in the same commit.
NON_TIERED_BENCHMARKS = {
    "when2call-eval",
    "tau-bench-tau2",
    "toolsandbox",
    "mcpmark-verified",
    "toolathlon",
    "internal-no-tool-regression",
}

# Claims the documentation/state audit removed. Each one contradicted an executed artifact, so a
# reappearance means the documentation has drifted back.
STALE_CLAIMS = (
    "three post-training",
    "four post-training",
    "DPO run and rejected",
    "Phase 1.0 Foundation",
    "arrogance231",
    "16-benchmark evaluation system",
    "OUT OF SCOPE — NOT ATTEMPTED",
)


def _registry_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _row(experiment_id: str) -> dict:
    rows = {row["experiment_id"]: row for row in _registry_rows()}
    assert experiment_id in rows, f"{experiment_id} missing from the derived registry"
    return rows[experiment_id]


def _record(experiment_id: str) -> dict:
    path = ROOT / "runs" / experiment_id / "experiment.json"
    assert path.is_file(), f"authoritative record missing: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _records() -> list[tuple[str, dict]]:
    found: list[tuple[str, dict]] = []
    for path in sorted((ROOT / "runs").rglob("experiment.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        found.append((str(payload.get("experiment_id", path.parent.name)), payload))
    return found


def test_mock_run_is_invalid_and_never_reads_as_trained() -> None:
    """The M2 mock pass must not be consumable as evidence of real distillation.

    Its ledger still holds the original TRAINED events (append-only history), so the correction
    has to live in the record status and an explicit validity annotation.
    """
    row = _row("qwen35_2b_m2_distill")
    assert row["status"] == "INVALID", "a mock-only run must not report a training status"
    assert row["headline_checkpoint_selection"] == "NO_EVALUATED_CHECKPOINT"
    record = _record("qwen35_2b_m2_distill")
    assert record["metadata"]["validity"] == "MOCK_ONLY"
    assert "reports/M2_DECISION.md" in record["metadata"]["validity_note"]


def test_no_mock_or_scaffold_record_carries_an_effective_claim() -> None:
    """Validity annotations gate every count, so an unannotated mock cannot slip through."""
    _items = list(_records())
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for experiment_id, record in _items:
        validity = (record.get("metadata") or {}).get("validity")
        if validity == "MOCK_ONLY":
            assert record["status"] not in {"TRAINED", "PROMOTED", "EVALUATED"}, (
                f"{experiment_id} is annotated MOCK_ONLY but claims {record['status']}"
            )
        if validity == "SCAFFOLD_ONLY":
            assert record["status"] != "INVALID", (
                f"{experiment_id} is scaffold-era history, not an invalid run"
            )


def test_scaffold_era_records_are_annotated_and_keep_their_history() -> None:
    """The scaffold identities keep their lifecycle facts but are excluded from real counts."""
    for experiment_id in ("qwen35_2b_m0_sft", "qwen35_2b_m1_dpo"):
        record = _record(experiment_id)
        assert record["metadata"]["validity"] == "SCAFFOLD_ONLY"
    assert _record("qwen35_2b_m0_sft")["status"] == "PROMOTED"
    assert _record("qwen35_2b_m1_dpo")["status"] == "TRAINED"
    # The repeat run is a reproduction attempt, not a distinct intervention.
    assert _record("qwen35_2b_m1_dpo_v1_restore")["metadata"]["validity"] == (
        "NON_REPRODUCIBLE_REPEAT"
    )


def test_every_valid_promotion_has_a_promoted_registered_checkpoint() -> None:
    """A promotion without a promoted checkpoint in the registry is an unsupported claim."""
    checkpoints = {
        entry["checkpoint_id"]: entry
        for entry in json.loads(CHECKPOINT_REGISTRY.read_text(encoding="utf-8"))["checkpoints"]
    }
    valid_promotions = [
        row
        for row in _registry_rows()
        if row["status"] == "PROMOTED"
        and (ROOT / "runs" / row["experiment_id"] / "experiment.json").is_file()
        and json.loads(
            (ROOT / "runs" / row["experiment_id"] / "experiment.json").read_text(encoding="utf-8")
        )["metadata"].get("validity")
        not in {"MOCK_ONLY", "SCAFFOLD_ONLY"}
    ]
    assert valid_promotions, "the promoted M1-v2 DPO should be a valid promotion"
    for row in valid_promotions:
        promoted = row["promotion_checkpoint"]
        assert promoted, f"{row['experiment_id']} is PROMOTED with no promotion_checkpoint"
        # The registry keys checkpoints by their full compound id (experiment::checkpoint).
        assert promoted in checkpoints, f"{promoted} is not in the checkpoint registry"
        assert checkpoints[promoted]["promotion_status"] == "PROMOTED"


def test_promoted_dpo_is_published_and_quotes_the_confirmatory_score() -> None:
    """The published-model claim in the documentation must trace to the release record.

    This is the guard against regressing to "DPO was rejected, no promoted model exists".
    """
    row = _row("m1_dpo_canonical_v2_final_v2")
    assert row["status"] == "PROMOTED"
    assert row["promotion_checkpoint"].endswith("dpo-checkpoint-30")

    release_path = (
        ROOT / "reports" / "releases" / "hf-publication-2026-09-11-m1-canonical-v2-final-v2.json"
    )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    assert release["experiment_id"] == "m1_dpo_canonical_v2_final_v2"
    assert release["repository"] == "arrochi112/OpenGrad-Qwen3.5-2B-M1-DPO-CanonicalV2-Final-v2"

    metrics_path = (
        ROOT
        / "runs"
        / "m1_dpo_canonical_v2_final_v2"
        / "eval"
        / "confirmatory"
        / "checkpoint-30"
        / "metrics.json"
    )
    comparison = json.loads(metrics_path.read_text(encoding="utf-8"))["baseline_comparison"]["metrics"]
    assert round(comparison["call_f1"]["candidate"], 4) == round(release["confirmatory"]["call_f1"], 4)
    assert round(comparison["call_recall"]["candidate"], 4) == round(
        release["confirmatory"]["call_recall"], 4
    )


def test_published_model_links_are_well_formed() -> None:
    """A published-artifact claim must name an owner/repository, not a free-text note."""
    pattern = re.compile(r"^[\w.-]+/[\w.-]+$")
    _items = sorted((ROOT / "reports" / "releases").glob("*.json"))
    assert _items, "nothing to check: an empty collection would pass this test vacuously"
    for path in _items:
        release = json.loads(path.read_text(encoding="utf-8"))
        candidates = [release.get("repository"), release.get("hub_repository")]
        candidates += [model.get("repository") for model in release.get("models") or []]
        candidates.append((release.get("model") or {}).get("repository"))
        for repository in candidates:
            if repository:
                assert pattern.match(str(repository)), f"{path.name}: bad repository {repository!r}"


def test_canonical_v2_dataset_figures_match_the_registry() -> None:
    """The counts the README and roadmap quote must come from the committed manifests."""
    datasets = {
        entry["id"]: entry
        for entry in yaml.safe_load(
            (ROOT / "registry" / "datasets.yaml").read_text(encoding="utf-8")
        )["datasets"]
    }
    final = datasets["canonical_v2_final"]
    assert final["retained_sample_count"] == 173_237
    assert final["checksum"].startswith("8ced403b")
    assert len(final["derived_from"]) == 4
    trainable = sum(
        kind["trainable_records"] for kind in final["supervision"]["by_kind"].values()
    )
    assert trainable == 161_966

    # The 103,036 snapshot is historical evidence, not the current corpus.
    assert datasets["canonical_v2"]["retained_sample_count"] == 103_036

    manifest = json.loads(
        (ROOT / "reports" / "evaluation" / "behavioral-heldout-v2.manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["deduplication"]["distinct_items"] == 3_652
    baseline = json.loads(
        (ROOT / "reports" / "baselines" / "qwen35_2b_baseline" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert baseline["records"] == 3_650


def test_benchmark_inventory_matches_the_documented_convention() -> None:
    """README's benchmark inventory is a claim about the registry and must stay true.

    The convention is 17 tiered external benchmarks plus the six documented non-tiered
    identifiers. Changing either number is a documentation change, not a silent test update.
    """
    benchmarks = yaml.safe_load(
        (ROOT / "registry" / "benchmarks.yaml").read_text(encoding="utf-8")
    )["benchmarks"]
    by_id = {entry["id"]: entry for entry in benchmarks}
    assert len(by_id) == 23

    tiered = {benchmark_id for benchmark_id, entry in by_id.items() if entry.get("tier")}
    assert len(tiered) == 17
    tiers = {entry["tier"] for entry in benchmarks if entry.get("tier")}
    assert tiers <= {"TIER_A", "TIER_B", "TIER_C", "TIER_D", "TIER_E"}

    non_tiered = set(by_id) - tiered
    assert non_tiered == NON_TIERED_BENCHMARKS, (
        "the non-tiered benchmark set changed; update README's counting convention too"
    )

    # Only the behavioral held-out has executed scores; every external family is frozen.
    assert "when2call-eval" in non_tiered
    for benchmark_id in tiered:
        assert by_id[benchmark_id]["prohibit_training"] is True


def test_generated_status_document_is_current() -> None:
    """The committed status table must equal a fresh regeneration from the artifacts."""
    committed = (ROOT / "docs" / "EXPERIMENT_STATUS.md").read_text(encoding="utf-8")
    assert committed == build_document(), (
        "docs/EXPERIMENT_STATUS.md is stale; run "
        "python scripts/reporting/generate_experiment_status.py"
    )
    # The generated view must agree with the counting convention it documents.
    assert "interventions with real model evidence: 7." in committed
    assert "Promoted models: 1." in committed


def test_registry_projection_is_not_stale() -> None:
    """The materialized index must agree with the artifacts it is derived from.

    ``validate_registry`` reports and never repairs, so the three codes below are the documented
    never-evaluated classes that already exist for scaffold, failed, and invalid runs (their
    ``eval/`` directories were never created). Any other code means the materialized index drifted
    from the authoritative records and must be rebuilt rather than tolerated.
    """
    documented_never_evaluated = {
        "STATUS_WITHOUT_EVAL_ARTIFACTS",
        "CURVE_POINT_WITHOUT_METRICS",
        "PROVENANCE_PATH_UNRESOLVED",
    }
    findings = validate_registry(ROOT)
    unexpected = [finding for finding in findings if finding.code not in documented_never_evaluated]
    assert unexpected == [], "\n".join(finding.render() for finding in unexpected)


def _heading_slug(text: str) -> str:
    """GitHub's heading anchor: lowercase, strip punctuation, then each whitespace becomes '-'."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def _markdown_anchors(path: Path) -> set[str]:
    anchors = set()
    if not path.is_file():
        return anchors
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            anchors.add(_heading_slug(heading.group(1)))
    return anchors


def test_documentation_links_and_anchors_resolve() -> None:
    """A moved section must not leave a dangling link behind.

    Checks every tracked Markdown file's relative link targets and heading anchors. Tracked files only
    (via ``git ls-files``) so untracked local notes cannot fail the suite.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.split()
    assert tracked, "git ls-files returned no Markdown files"

    link_re = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
    problems: list[str] = []
    for relative in tracked:
        path = ROOT / relative
        if path.name.lower().startswith("readme.template"):
            continue
        for target in link_re.findall(path.read_text(encoding="utf-8", errors="replace")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            path_part, _, anchor = target.partition("#")
            if not path_part:
                if anchor and _heading_slug(anchor) not in _markdown_anchors(path):
                    problems.append(f"{relative} -> #{anchor}")
                continue
            resolved = (path.parent / path_part).resolve()
            broken_target = not resolved.exists()
            broken_anchor = (
                not broken_target
                and bool(anchor)
                and resolved.suffix == ".md"
                and _heading_slug(anchor) not in _markdown_anchors(resolved)
            )
            if broken_target or broken_anchor:
                problems.append(f"{relative} -> {target}")
    assert problems == [], "broken documentation links:\n" + "\n".join(sorted(set(problems)))


def test_removed_stale_claims_do_not_return() -> None:
    """Sentinel phrases stay deleted from the two documents a new reader starts from."""
    for name in ("README.md", "ROADMAP.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for claim in STALE_CLAIMS:
            assert claim not in text, f"{name} reintroduced the stale claim: {claim!r}"

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    # And the replacement story must be present rather than the claim merely deleted.
    assert "M1-v2" in readme and "promoted" in readme.lower()
    assert "EXPERIMENT_STATUS.md" in readme
