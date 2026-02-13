from __future__ import annotations

import scripts.run_validation_suite as rvs


def _base_report(pytest_rc: int) -> dict:
    return {
        "commands": {
            "validate_results_schema": {
                "returncode": 0,
            },
            "pytest_tier": {
                "returncode": pytest_rc,
            },
        }
    }


def test_extract_json_payload_from_plain_json() -> None:
    payload = rvs._extract_json_payload(
        '{"overall_status":"good","opencl":{},"recommendations":[]}'
    )
    assert payload is not None
    assert payload["overall_status"] == "good"


def test_extract_json_payload_with_prefixed_text() -> None:
    text = (
        "Running driver diagnostics...\n\n"
        '{"overall_status":"warning","opencl":{},"recommendations":["x"]}'
    )
    payload = rvs._extract_json_payload(text)
    assert payload is not None
    assert payload["overall_status"] == "warning"
    assert payload["recommendations"] == ["x"]


def test_evaluate_accepts_pytest_exit_code_5_when_allowed() -> None:
    report = _base_report(pytest_rc=5)
    evaluation = rvs._evaluate(report, allow_no_tests=True)

    assert evaluation["decision"] == "promote"
    assert evaluation["failed_checks"] == []
    assert evaluation["checks"]["pytest_tier_green"]["pass"] is True
    assert evaluation["checks"]["pytest_tier_green"]["allow_no_tests"] is True


def test_evaluate_rejects_pytest_exit_code_5_when_not_allowed() -> None:
    report = _base_report(pytest_rc=5)
    evaluation = rvs._evaluate(report, allow_no_tests=False)

    assert evaluation["decision"] == "iterate"
    assert "pytest_tier_green" in evaluation["failed_checks"]
    assert evaluation["checks"]["pytest_tier_green"]["pass"] is False


def test_evaluate_smoke_json_requires_expected_keys() -> None:
    report = _base_report(pytest_rc=0)
    report["commands"]["verify_drivers_smoke"] = {
        "returncode": 1,
        "json_parse_ok": True,
        "parsed_keys": ["overall_status", "opencl", "recommendations"],
        "overall_status": "error",
    }

    evaluation = rvs._evaluate(report, allow_no_tests=True)
    assert evaluation["decision"] == "promote"
    assert evaluation["checks"]["verify_drivers_json_smoke"]["pass"] is True

    report["commands"]["verify_drivers_smoke"]["parsed_keys"] = ["overall_status", "opencl"]
    evaluation_missing = rvs._evaluate(report, allow_no_tests=True)
    assert evaluation_missing["decision"] == "iterate"
    assert "verify_drivers_json_smoke" in evaluation_missing["failed_checks"]


def test_evaluate_accepts_skipped_pytest_by_flag() -> None:
    report = _base_report(pytest_rc=0)
    report["commands"]["pytest_tier"]["skipped_by_flag"] = True

    evaluation = rvs._evaluate(report, allow_no_tests=False)
    assert evaluation["decision"] == "promote"
    assert evaluation["checks"]["pytest_tier_green"]["pass"] is True
