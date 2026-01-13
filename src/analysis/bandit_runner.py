# -*- coding: utf-8 -*-
"""
Bandit runner (Step A1)

Goal:
  Run Bandit on:
    - a Python file
    - a directory (recursive)
    - a Python code string (written to a temp file)
  and return a structured report.

Dependencies:
  pip install bandit

Notes:
  - Bandit may return non-zero exit codes when issues are found.
    We still parse stdout JSON in that case.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path


@dataclass
class BanditIssue:
    test_id: str
    test_name: str
    issue_text: str
    severity: str
    confidence: str
    filename: str
    line_number: int
    line_range: List[int]
    code: str
    cwe_id: Optional[int] = None
    cwe_link: Optional[str] = None
    more_info: Optional[str] = None


@dataclass
class BanditReport:
    generated_at: Optional[str]
    errors: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    issues: List[BanditIssue]
    raw: Dict[str, Any]


class BanditRunnerError(RuntimeError):
    pass


def _guess_bandit_command() -> List[str]:
    """
    Prefer `python -m bandit` to avoid PATH issues.
    """
    return [os.environ.get("PYTHON", "python"), "-m", "bandit"]


def _run_subprocess(cmd: Sequence[str], timeout_s: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as e:
        raise BanditRunnerError(
            f"Cannot execute Bandit command. Tried: {cmd}. "
            f"Ensure `bandit` is installed: `pip install bandit`."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise BanditRunnerError(f"Bandit timed out after {timeout_s}s. cmd={cmd}") from e


def _parse_bandit_json(stdout: str, stderr: str, cmd: Sequence[str]) -> Dict[str, Any]:
    stdout = stdout.strip()
    if not stdout:
        raise BanditRunnerError(
            "Bandit produced empty stdout; cannot parse JSON. "
            f"cmd={cmd}\n"
            f"stderr:\n{stderr.strip()}"
        )
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise BanditRunnerError(
            "Failed to parse Bandit JSON output.\n"
            f"cmd={cmd}\n"
            f"stderr:\n{stderr.strip()}\n"
            f"stdout(head):\n{stdout[:500]}"
        ) from e


def _normalize_issue(item: Dict[str, Any]) -> BanditIssue:
    cwe = item.get("cwe") or {}
    cwe_id = cwe.get("id") if isinstance(cwe, dict) else None
    cwe_link = cwe.get("link") if isinstance(cwe, dict) else None

    # Some Bandit versions store code in `code`, some in `code` with indentation preserved
    code = item.get("code") or ""
    if isinstance(code, str):
        code = code.rstrip("\n")

    line_range = item.get("line_range") or []
    if not isinstance(line_range, list):
        line_range = []

    return BanditIssue(
        test_id=item.get("test_id", ""),
        test_name=item.get("test_name", ""),
        issue_text=item.get("issue_text", ""),
        severity=item.get("issue_severity", ""),
        confidence=item.get("issue_confidence", ""),
        filename=item.get("filename", ""),
        line_number=int(item.get("line_number", 0) or 0),
        line_range=[int(x) for x in line_range if isinstance(x, (int, float, str)) and str(x).isdigit()],
        code=code,
        cwe_id=int(cwe_id) if isinstance(cwe_id, (int, float)) else None,
        cwe_link=str(cwe_link) if cwe_link else None,
        more_info=item.get("more_info"),
    )


def run_bandit_on_path(
    target_path: Union[str, Path],
    *,
    recursive: Optional[bool] = None,
    severity_level: str = "LOW",
    confidence_level: str = "LOW",
    exclude: Optional[Sequence[str]] = None,
    config_file: Optional[Union[str, Path]] = None,
    timeout_s: int = 120,
) -> BanditReport:
    """
    Run Bandit on a file or directory.

    Args:
        target_path: file or directory
        recursive: if None, auto: True for directory, False for file
        severity_level: LOW/MEDIUM/HIGH
        confidence_level: LOW/MEDIUM/HIGH
        exclude: patterns/paths to exclude
        config_file: bandit config yaml/ini
        timeout_s: subprocess timeout

    Returns:
        BanditReport
    """
    target_path = Path(target_path)
    if not target_path.exists():
        raise BanditRunnerError(f"Target path does not exist: {target_path}")

    if recursive is None:
        recursive = target_path.is_dir()

    cmd = _guess_bandit_command()
    cmd += ["-f", "json"]

    # quiet reduces banner noise in some versions; safe even if ignored
    cmd += ["-q"]

    cmd += ["--severity-level", severity_level]
    cmd += ["--confidence-level", confidence_level]

    if exclude:
        # Bandit expects a comma-separated string for --exclude
        cmd += ["--exclude", ",".join(map(str, exclude))]

    if config_file:
        cmd += ["-c", str(config_file)]

    if recursive:
        cmd += ["-r", str(target_path)]
    else:
        cmd += [str(target_path)]

    proc = _run_subprocess(cmd, timeout_s=timeout_s)

    data = _parse_bandit_json(proc.stdout, proc.stderr, cmd)

    issues = [_normalize_issue(x) for x in (data.get("results") or [])]
    report = BanditReport(
        generated_at=data.get("generated_at"),
        errors=data.get("errors") or [],
        metrics=data.get("metrics") or {},
        issues=issues,
        raw=data,
    )
    return report


def run_bandit_on_code(
    code: str,
    *,
    filename_hint: str = "snippet.py",
    severity_level: str = "LOW",
    confidence_level: str = "LOW",
    config_file: Optional[Union[str, Path]] = None,
    timeout_s: int = 120,
) -> BanditReport:
    """
    Run Bandit on a Python code string by writing to a temp file.
    """
    if not isinstance(code, str) or not code.strip():
        raise BanditRunnerError("Empty code string provided to run_bandit_on_code().")

    with tempfile.TemporaryDirectory(prefix="bandit_tmp_") as td:
        tmp_path = Path(td) / filename_hint
        tmp_path.write_text(code, encoding="utf-8")
        return run_bandit_on_path(
            tmp_path,
            recursive=False,
            severity_level=severity_level,
            confidence_level=confidence_level,
            exclude=None,
            config_file=config_file,
            timeout_s=timeout_s,
        )


def report_to_jsonable(report: BanditReport) -> Dict[str, Any]:
    """
    Convert BanditReport to a JSON-serializable dict (dataclass -> dict).
    """
    d = asdict(report)
    # issues are already dataclasses converted by asdict
    return d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Bandit and print JSON issues.")
    parser.add_argument("target", help="Path to .py file or directory")
    parser.add_argument("--recursive", action="store_true", help="Force recursive scan")
    parser.add_argument("--severity", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--confidence", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--config", default=None, help="Bandit config file path")
    parser.add_argument("--exclude", default=None, help="Comma-separated exclude paths/patterns")
    args = parser.parse_args()

    exclude = args.exclude.split(",") if args.exclude else None
    rep = run_bandit_on_path(
        args.target,
        recursive=args.recursive if args.recursive else None,
        severity_level=args.severity,
        confidence_level=args.confidence,
        exclude=exclude,
        config_file=args.config,
    )
    print(json.dumps(report_to_jsonable(rep), ensure_ascii=False, indent=2))

