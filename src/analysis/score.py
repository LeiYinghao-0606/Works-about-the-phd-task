# -*- coding: utf-8 -*-
"""
Output per sample:
  - focus_lines: lines to focus for subgraph extraction / edit mask
  - risk_score: continuous score for guidance (larger = riskier)
  - cond_tags: tokens for conditioning (CWE/test_id/sev/conf)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .bandit_runner import BanditReport, BanditIssue
from .cwe_mapper import build_issue_tags, aggregate_tag_counts


_SEV_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_CONF_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# You can tune these weights later (important for guidance strength).
_DEFAULT_SEV_W = {"LOW": 1.0, "MEDIUM": 3.0, "HIGH": 9.0}
_DEFAULT_CONF_W = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 3.0}


@dataclass
class SecuritySignals:
    safe: bool
    issue_count: int
    risk_score: float
    risk_score_log: float
    focus_lines: List[int]
    cond_tags: List[str]
    tag_counts: Dict[str, int]
    # optional: keep lightweight per-issue summaries for debugging
    issue_summaries: List[Dict[str, Any]]


def _norm_level(x: str, allowed: Set[str], default: str) -> str:
    x = (x or "").strip().upper()
    return x if x in allowed else default


def _passes_threshold(
    sev: str,
    conf: str,
    min_severity: str,
    min_confidence: str,
) -> bool:
    sev = _norm_level(sev, {"LOW", "MEDIUM", "HIGH"}, "LOW")
    conf = _norm_level(conf, {"LOW", "MEDIUM", "HIGH"}, "LOW")
    return (_SEV_RANK[sev] >= _SEV_RANK[min_severity]) and (_CONF_RANK[conf] >= _CONF_RANK[min_confidence])


def _issue_weight(
    issue: BanditIssue,
    sev_w: Dict[str, float],
    conf_w: Dict[str, float],
) -> float:
    sev = _norm_level(issue.severity, {"LOW", "MEDIUM", "HIGH"}, "LOW")
    conf = _norm_level(issue.confidence, {"LOW", "MEDIUM", "HIGH"}, "LOW")
    return float(sev_w.get(sev, 1.0)) * float(conf_w.get(conf, 1.0))


def _collect_focus_lines(issue: BanditIssue, context_window: int) -> Set[int]:
    """
    focus lines = (line_range if available else line_number) plus +/- context_window
    """
    lines: Set[int] = set()

    base: List[int] = []
    if issue.line_range:
        base = [x for x in issue.line_range if isinstance(x, int) and x > 0]
    if not base and issue.line_number and issue.line_number > 0:
        base = [int(issue.line_number)]

    for ln in base:
        for k in range(-context_window, context_window + 1):
            v = ln + k
            if v > 0:
                lines.add(v)
    return lines


def extract_security_signals(
    report: BanditReport,
    *,
    min_severity: str = "LOW",
    min_confidence: str = "LOW",
    context_window: int = 0,
    sev_weights: Optional[Dict[str, float]] = None,
    conf_weights: Optional[Dict[str, float]] = None,
    topk_tags: int = 8,
) -> SecuritySignals:
    """
    Aggregate bandit report to security signals.

    Args:
        min_severity/min_confidence: filter issues for scoring and focus line extraction
        context_window: expand focus lines by +/- window
        topk_tags: keep top-k tags as conditioning tokens (ordered by weighted frequency)

    Returns:
        SecuritySignals
    """
    min_severity = _norm_level(min_severity, {"LOW", "MEDIUM", "HIGH"}, "LOW")
    min_confidence = _norm_level(min_confidence, {"LOW", "MEDIUM", "HIGH"}, "LOW")
    sev_w = sev_weights or _DEFAULT_SEV_W
    conf_w = conf_weights or _DEFAULT_CONF_W

    kept: List[BanditIssue] = []
    for it in report.issues:
        if _passes_threshold(it.severity, it.confidence, min_severity, min_confidence):
            kept.append(it)

    # risk_score: sum of per-issue weights (severity x confidence)
    raw_score = 0.0
    focus: Set[int] = set()
    issue_summaries: List[Dict[str, Any]] = []

    for it in kept:
        w = _issue_weight(it, sev_w, conf_w)
        raw_score += w
        focus |= _collect_focus_lines(it, context_window=context_window)

        tags = build_issue_tags(it).tags
        issue_summaries.append(
            {
                "test_id": it.test_id,
                "cwe_id": it.cwe_id,
                "severity": it.severity,
                "confidence": it.confidence,
                "line": it.line_number,
                "weight": w,
                "tags": tags,
                "text": it.issue_text,
            }
        )

    # log score is often more numerically stable for downstream scaling
    log_score = math.log1p(raw_score)

    # tags (conditioning tokens)
    tag_counts = aggregate_tag_counts(kept)

    # To choose top-k tags, we weight tag frequency by issue weights to emphasize severe ones.
    weighted_counts: Dict[str, float] = {}
    for it in kept:
        w = _issue_weight(it, sev_w, conf_w)
        for t in build_issue_tags(it).tags:
            weighted_counts[t] = weighted_counts.get(t, 0.0) + w

    sorted_tags = sorted(weighted_counts.items(), key=lambda x: (-x[1], x[0]))
    cond_tags = [t for t, _ in sorted_tags[: max(0, int(topk_tags))]]

    safe = (len(kept) == 0)

    return SecuritySignals(
        safe=safe,
        issue_count=len(kept),
        risk_score=float(raw_score),
        risk_score_log=float(log_score),
        focus_lines=sorted(focus),
        cond_tags=cond_tags,
        tag_counts=tag_counts,
        issue_summaries=issue_summaries,
    )


# Optional CLI for quick debugging
if __name__ == "__main__":
    import json
    import argparse
    from .bandit_runner import run_bandit_on_path, report_to_jsonable

    parser = argparse.ArgumentParser(description="Run Bandit and aggregate security signals.")
    parser.add_argument("target", help="Path to .py file or directory")
    parser.add_argument("--min_sev", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--min_conf", default="LOW", choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--ctx", type=int, default=0, help="focus line context window")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    rep = run_bandit_on_path(args.target)
    sig = extract_security_signals(
        rep,
        min_severity=args.min_sev,
        min_confidence=args.min_conf,
        context_window=args.ctx,
        topk_tags=args.topk,
    )

    out = {
        "signals": asdict(sig),
        "bandit_meta": {"generated_at": rep.generated_at, "errors": rep.errors, "metrics": rep.metrics},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

