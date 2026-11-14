"""Run the full test suite and write docs/test_report.md (counts, per-test results, acceptance).

    python scripts/run_tests.py
Exit code is non-zero if any test failed.
"""
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNIT = ROOT / "reports" / "tests" / "junit.xml"


def main() -> int:
    JUNIT.parent.mkdir(parents=True, exist_ok=True)
    code = subprocess.call([sys.executable, "-m", "pytest", "-q", f"--junitxml={JUNIT}"], cwd=ROOT)
    suite = ET.parse(JUNIT).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    cases = []
    for tc in suite.iter("testcase"):
        status = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "FAILED"
        elif tc.find("skipped") is not None:
            status = "skipped"
        cases.append((tc.get("classname", "").split(".")[-1], tc.get("name"), status, float(tc.get("time", 0))))
    total = len(cases)
    passed = sum(c[2] == "passed" for c in cases)
    failed = sum(c[2] == "FAILED" for c in cases)
    skipped = total - passed - failed

    lines = ["# Test Report", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M} by `python scripts/run_tests.py`", "",
             "## Summary", "", "| Total | Passed | Failed | Skipped | Duration (s) |",
             "|---|---|---|---|---|",
             f"| {total} | {passed} | {failed} | {skipped} | {sum(c[3] for c in cases):.1f} |", "",
             "## Results by module", "", "| Module | Tests | Passed | Failed |", "|---|---|---|---|"]
    modules = sorted({c[0] for c in cases})
    for mod in modules:
        mc = [c for c in cases if c[0] == mod]
        lines.append(f"| {mod} | {len(mc)} | {sum(c[2] == 'passed' for c in mc)} | "
                     f"{sum(c[2] == 'FAILED' for c in mc)} |")
    lines += ["", "## Acceptance criteria", "", "| Criterion | Required | Actual | Result |", "|---|---|---|---|",
              f"| All automated tests pass | 100 % | {100 * passed / max(total, 1):.0f} % | "
              f"{'PASS' if failed == 0 else 'FAIL'} |"]
    acc = ROOT / "reports" / "acceptance.json"
    if acc.exists():
        for name, c in json.loads(acc.read_text())["checks"].items():
            lines.append(f"| Model {name.replace('min_', '')} on test set | >= {c['required']} | "
                         f"{c['actual']} | {'PASS' if c['passed'] else 'FAIL'} |")
    else:
        lines.append("| Model metrics | see params.yaml | not available (run `dvc repro`) | - |")
    lines += ["", "## All test cases", "", "| Module | Test | Result | Time (s) |", "|---|---|---|---|"]
    lines += [f"| {m} | `{n}` | {s} | {t:.2f} |" for m, n, s, t in cases]
    (ROOT / "docs" / "test_report.md").write_text("\n".join(lines) + "\n")
    print(f"\n{total} tests: {passed} passed, {failed} failed, {skipped} skipped -> docs/test_report.md")
    return code


if __name__ == "__main__":
    sys.exit(main())
