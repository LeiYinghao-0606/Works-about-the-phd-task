# -*- coding: utf-8 -*-
"""
Design principle:
  - Trust Bandit's own CWE field when present (issue.cwe_id).
  - If CWE is missing, fall back to Bandit's test_id (e.g., "B608") as a stable tag.
  - Output "condition tags" that downstream models can use as conditioning tokens.

This avoids brittle hand-written mappings from Bandit rule -> CWE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .bandit_runner import BanditIssue


@dataclass(frozen=True)
class IssueTags:
    """
    tags: tokens for conditioning, e.g. ["CWE-089", "B608", "SEV-HIGH", "CONF-MEDIUM"]
    cwe_tag: "CWE-xxx" if available
    test_tag: "Bxxx" always available for Bandit
    """
    tags: List[str]
    cwe_tag: Optional[str]
    test_tag: str


def _normalize_severity(sev: str) -> str:
    sev = (sev or "").strip().upper()
    if sev not in {"LOW", "MEDIUM", "HIGH"}:
        sev = "LOW"
    return sev


def _normalize_confidence(conf: str) -> str:
    conf = (conf or "").strip().upper()
    if conf not in {"LOW", "MEDIUM", "HIGH"}:
        conf = "LOW"
    return conf


def issue_to_cwe_tag(issue: BanditIssue) -> Optional[str]:
    """
    Return "CWE-xxx" if Bandit provides cwe_id, else None.
    """
    if issue.cwe_id is None:
        return None
    # CWE id is usually int like 89; keep 3-digit zero padding for consistent tokens.
    return f"CWE-{int(issue.cwe_id):03d}"


def issue_to_test_tag(issue: BanditIssue) -> str:
    """
    Bandit test id is stable (e.g., "B608"). Use as fallback tag.
    """
    tid = (issue.test_id or "").strip().upper()
    return tid if tid else "B000"


def build_issue_tags(
    issue: BanditIssue,
    *,
    include_sev_conf_tags: bool = True,
    include_cwe_tag: bool = True,
    include_test_tag: bool = True,
) -> IssueTags:
    """
    Convert a single issue into conditioning tags.
    """
    tags: List[str] = []

    cwe_tag = issue_to_cwe_tag(issue)
    test_tag = issue_to_test_tag(issue)

    if include_cwe_tag and cwe_tag is not None:
        tags.append(cwe_tag)
    if include_test_tag and test_tag is not None:
        tags.append(test_tag)

    if include_sev_conf_tags:
        sev = _normalize_severity(issue.severity)
        conf = _normalize_confidence(issue.confidence)
        tags.append(f"SEV-{sev}")
        tags.append(f"CONF-{conf}")

    # You can add more tags here if you want:
    # - rule family tags
    # - project/package tags
    # - sink/source tags, etc.

    return IssueTags(tags=tags, cwe_tag=cwe_tag, test_tag=test_tag)


def aggregate_tag_counts(issues: Sequence[BanditIssue]) -> Dict[str, int]:
    """
    Count tags across issues (default tags include CWE/test_id/severity/confidence).
    """
    counts: Dict[str, int] = {}
    for it in issues:
        it_tags = build_issue_tags(it).tags
        for t in it_tags:
            counts[t] = counts.get(t, 0) + 1
    return counts

