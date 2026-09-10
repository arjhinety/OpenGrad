from opengrad.benchmarks.contamination.scanner import MultiLevelContaminationScanner


def test_contamination_scanner_exact_hash_match() -> None:
    scanner = MultiLevelContaminationScanner()
    bm_samples = [
        {"id": "b1", "prompt": "Find all users in database.", "canonical": "canon_exact_match"}
    ]
    tr_samples = [
        {"id": "t1", "prompt": "Find all users in database.", "canonical": "canon_exact_match"}
    ]

    report = scanner.scan("test_bm", bm_samples, tr_samples, max_level=5)
    assert report.verdict == "CONTAMINATED"
    assert report.matches_by_level[1] == 1
    assert len(report.audit_queue) == 1
    assert report.audit_queue[0].level == 1


def test_contamination_scanner_normalized_prompt_match() -> None:
    scanner = MultiLevelContaminationScanner()
    # Different whitespace / capitalization
    bm_samples = [{"id": "b1", "prompt": "   FIND ALL   USERS IN DB!  ", "canonical": "c1"}]
    tr_samples = [{"id": "t1", "prompt": "find all users in db!", "canonical": "c2"}]

    report = scanner.scan("test_bm", bm_samples, tr_samples, max_level=5)
    assert report.verdict == "CONTAMINATED"
    assert report.matches_by_level[2] == 1
    assert report.audit_queue[0].level == 2


def test_contamination_scanner_near_duplicate_and_audit_queue() -> None:
    scanner = MultiLevelContaminationScanner(ngram_threshold=0.75, semantic_threshold=0.80)
    bm_samples = [
        {
            "id": "b1",
            "prompt": "Could you please tell me the exact weather forecast for San Francisco tomorrow morning?",
            "canonical": "c1",
        }
    ]
    tr_samples = [
        {
            "id": "t1",
            "prompt": "Can you please tell me the exact weather forecast for San Francisco tomorrow morning?",
            "canonical": "c2",
        }
    ]

    report = scanner.scan("test_bm", bm_samples, tr_samples, max_level=5)
    assert report.verdict == "SUSPICIOUS_MATCHES"
    assert len(report.audit_queue) >= 1
    md = report.render_markdown()
    assert "Suspicious Matches in Audit Queue" in md
    assert "Policy Warning" in md
