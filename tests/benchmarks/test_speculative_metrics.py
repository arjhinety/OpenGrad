from opengrad.benchmarks.schema import NormalizedTaskResult
from opengrad.benchmarks.speculative.metrics import (
    compute_mtp_depth_metrics,
    compute_speedup,
)
from opengrad.benchmarks.speculative.pareto import (
    ParetoPoint,
    analyze_pareto,
    render_pareto_table,
)
from opengrad.benchmarks.speculative.parity import (
    check_tool_call_equivalence,
    evaluate_quality_parity,
)


def test_speedup_accounting() -> None:
    accounting = compute_speedup(
        ar_tokens_per_sec=50.0,
        spec_tokens_per_sec=95.0,
        ar_wall_time=2.0,
        spec_wall_time=1.05,
        ar_ttft_ms=15.0,
        spec_ttft_ms=16.0,
        ar_p95_itl_ms=10.0,
        spec_p95_itl_ms=8.0,
        ar_vram_mb=1000.0,
        spec_vram_mb=1400.0,
    )
    assert accounting.throughput_speedup == 1.9
    assert accounting.latency_speedup == round(2.0 / 1.05, 3)
    assert accounting.ttft_delta_ms == 1.0
    assert accounting.vram_delta_mb == 400.0


def test_mtp_depth_metrics() -> None:
    # Depth 1: 85% accepted, Depth 2: 70% accepted, Depth 3: 40% accepted, Depth 4: 15% accepted
    counts = {
        1: (85, 100),
        2: (70, 100),
        3: (40, 100),
        4: (15, 100),
    }
    mtp = compute_mtp_depth_metrics(counts)
    assert mtp.per_depth_acceptance[1] == 0.85
    assert mtp.per_depth_acceptance[4] == 0.15
    # Useful depth is 3 because depth 4 drops below 30% threshold
    assert mtp.useful_speculation_depth == 3
    assert mtp.wasted_speculative_computation_pct > 0.0


def test_quality_parity_evaluation() -> None:
    ar_task = NormalizedTaskResult(
        task_id="t1",
        input="call",
        raw_output="<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"1\"}}</tool_call>",
        parsed_output={"decision": "CALL", "name": "lookup", "arguments": {"q": "1"}},
        expected={},
        score=1.0,
        success=True,
        tool_calls=[{"name": "lookup", "arguments": {"q": "1"}}],
    )
    spec_task_exact = NormalizedTaskResult(
        task_id="t1",
        input="call",
        raw_output="<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"1\"}}</tool_call>",
        parsed_output={"decision": "CALL", "name": "lookup", "arguments": {"q": "1"}},
        expected={},
        score=1.0,
        success=True,
        tool_calls=[{"name": "lookup", "arguments": {"q": "1"}}],
    )
    equiv = check_tool_call_equivalence(ar_task, spec_task_exact)
    assert equiv.is_equivalent is True

    summary = evaluate_quality_parity([ar_task], [spec_task_exact])
    assert summary.parity_maintained is True
    assert summary.exact_token_match_rate == 1.0
    assert summary.score_delta == 0.0


def test_pareto_analysis() -> None:
    points = [
        ParetoPoint("Config_A", speedup=1.4, quality_delta=-0.05, benchmark_score=72.2),
        ParetoPoint("Config_B", speedup=1.8, quality_delta=-0.1, benchmark_score=72.1),
        ParetoPoint("Config_C_Dominated", speedup=1.3, quality_delta=-0.2, benchmark_score=72.0),
        ParetoPoint("Config_D_Unacceptable", speedup=2.1, quality_delta=-4.5, benchmark_score=67.7),
    ]
    analyzed = analyze_pareto(points, max_acceptable_quality_drop=1.0)

    # Config_C should be dominated by Config_A and Config_B
    pt_c = next(p for p in analyzed if p.config_name == "Config_C_Dominated")
    assert pt_c.dominated is True
    assert "Config_A" in pt_c.dominated_by or "Config_B" in pt_c.dominated_by

    # Config_D should be flagged unacceptable due to -4.5 drop
    pt_d = next(p for p in analyzed if p.config_name == "Config_D_Unacceptable")
    assert pt_d.acceptable is False

    table = render_pareto_table(analyzed)
    assert "Config_A" in table
    assert "REJECT" in table
