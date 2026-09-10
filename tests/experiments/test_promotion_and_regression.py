from opengrad.failures.analyzer import FailureAnalyzer, FailureItem
from opengrad.promotion.policy import PromotionDecision, PromotionPolicy
from opengrad.promotion.regression import RegressionEngine


def test_regression_engine_detects_drop() -> None:
    engine = RegressionEngine(regression_threshold=1.0)
    baseline = {"bfcl-v4": 72.0, "ifeval": 85.0}
    candidate = {"bfcl-v4": 75.0, "ifeval": 81.0}  # IFEval dropped by 4.0

    report = engine.compare(baseline, candidate, "base_01", "cand_01")
    assert report.total_regressions == 1
    assert report.total_improvements == 1
    assert report.verdict == "FAIL"


def test_promotion_policy_evaluation() -> None:
    engine = RegressionEngine()
    baseline = {"bfcl-v4": 70.0, "ifeval": 80.0}
    candidate = {"bfcl-v4": 74.0, "ifeval": 79.5}  # IFEval drop within 1.0 tolerance

    report = engine.compare(baseline, candidate)
    policy = PromotionPolicy(
        must_pass={"bfcl-v4": 65.0},
        max_regression={"default": 1.0},
        min_improvement={"bfcl-v4": 2.0},
    )
    verdict = policy.evaluate(report, candidate)
    assert verdict.decision == PromotionDecision.PROMOTE.value


def test_failure_clustering_and_diff() -> None:
    analyzer = FailureAnalyzer()
    f1 = FailureItem("bfcl", "s1", "call", {}, {}, 0.0, "wrong_tool", "c1", "e1")
    f2 = FailureItem("bfcl", "s2", "call", {}, {}, 0.0, "wrong_tool", "c1", "e1")
    f3 = FailureItem("bfcl", "s3", "call", {}, {}, 0.0, "missed_tool", "c1", "e1")

    clusters = analyzer.cluster([f1, f2, f3])
    assert len(clusters) == 2
    assert clusters[0].category == "wrong_tool"
    assert clusters[0].count == 2

    # Test failure diff
    diff = analyzer.diff(
        baseline_failures=[f1],
        candidate_failures=[f2, f3],
        baseline_id="base",
        candidate_id="cand",
    )
    assert len(diff.new_failures) == 2
    assert len(diff.fixed_failures) == 1
