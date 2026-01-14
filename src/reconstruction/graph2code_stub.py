# -*- coding: utf-8 -*-
"""
Goal:
  Convert a "guided / improved graph" candidate into an executable code patch,
  so we can run external verifier (Bandit/Semgrep/CodeQL).

Design:
  - Use graph to localize edits:
      * focus_nodes + lineno, or edit_mask + lineno
  - Use cond_tags (e.g., Bandit test IDs like B506/B602/B307/...) to select patch rules.
  - Apply conservative, local, syntactic-safe transformations where patterns match.

Limitations:
  - This is a stub baseline. It cannot reconstruct arbitrary edits from node_type alone.
  - For stronger performance, replace rule actions with an LLM patch generator (still using
    the same "localization + tag-conditioned repair" interface).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Sequence, Any
import re


# -------------------------
# Report structure
# -------------------------

@dataclass
class PatchEdit:
    rule: str
    line_no: int
    before: str
    after: str


@dataclass
class PatchReport:
    applied_rules: List[str] = field(default_factory=list)
    applied_tags: List[str] = field(default_factory=list)
    edits: List[PatchEdit] = field(default_factory=list)
    touched_lines: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# -------------------------
# Helpers: localization
# -------------------------

def _get_lines_to_patch(
    x0_graph: Any,
    x1_graph: Optional[Any] = None,
    *,
    context_window: int = 2,
    max_lines: int = 80,
) -> List[int]:
    """
    Determine candidate source line numbers to patch.

    Priority:
      1) focus_nodes (if exists) -> lineno
      2) edit_mask == True nodes -> lineno
    Fallback:
      none => empty list
    """
    lines: List[int] = []

    def _collect_from_focus(g: Any) -> List[int]:
        out = []
        if hasattr(g, "focus_nodes") and hasattr(g, "lineno"):
            idx = getattr(g, "focus_nodes")
            # idx could be tensor or list
            try:
                idx_list = idx.tolist()
            except Exception:
                idx_list = list(idx)
            for i in idx_list:
                try:
                    ln = int(g.lineno[int(i)])
                except Exception:
                    ln = 0
                if ln > 0:
                    out.append(ln)
        return out

    def _collect_from_edit(g: Any) -> List[int]:
        out = []
        if hasattr(g, "edit_mask") and hasattr(g, "lineno"):
            m = getattr(g, "edit_mask")
            try:
                m_list = m.tolist()
            except Exception:
                m_list = list(m)
            for i, flag in enumerate(m_list):
                if bool(flag):
                    try:
                        ln = int(g.lineno[int(i)])
                    except Exception:
                        ln = 0
                    if ln > 0:
                        out.append(ln)
        return out

    lines.extend(_collect_from_focus(x0_graph))
    if not lines:
        lines.extend(_collect_from_edit(x0_graph))
    if not lines and x1_graph is not None:
        # sometimes x0 might not carry focus_nodes after serialization; try x1
        lines.extend(_collect_from_focus(x1_graph))
        if not lines:
            lines.extend(_collect_from_edit(x1_graph))

    if not lines:
        return []

    # add context window
    line_set = set()
    for ln in lines:
        for t in range(max(1, ln - context_window), ln + context_window + 1):
            line_set.add(t)

    # clamp size
    out = sorted(line_set)
    if len(out) > max_lines:
        # keep a centered subset
        mid = out[len(out) // 2]
        out = [x for x in out if abs(x - mid) <= max_lines // 2]
    return out


def _extract_tags(g: Any) -> List[str]:
    """
    Extract condition tags stored in Data. We expect g.cond_tags to be a list[str].
    """
    if hasattr(g, "cond_tags"):
        try:
            tags = list(getattr(g, "cond_tags"))
            return [str(t) for t in tags]
        except Exception:
            pass
    return []


# -------------------------
# Helpers: import insertion
# -------------------------

def _has_import(lines: List[str], needle: str) -> bool:
    pat = re.compile(r"^\s*(import|from)\s+")
    for s in lines[:200]:
        if pat.search(s) and needle in s:
            return True
    return False


def _insert_import(lines: List[str], import_stmt: str) -> List[str]:
    """
    Insert an import statement after shebang/encoding/module docstring/import block.
    Conservative placement to avoid breaking formatting.
    """
    if any(import_stmt.strip() == ln.strip() for ln in lines):
        return lines

    # skip shebang / encoding
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    if i < len(lines) and "coding" in lines[i]:
        i += 1

    # skip leading blank lines
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # skip module docstring if present
    if i < len(lines) and (lines[i].lstrip().startswith('"""') or lines[i].lstrip().startswith("'''")):
        quote = '"""' if lines[i].lstrip().startswith('"""') else "'''"
        i += 1
        while i < len(lines):
            if quote in lines[i]:
                i += 1
                break
            i += 1

    # skip existing imports
    while i < len(lines) and re.match(r"^\s*(import|from)\s+", lines[i]):
        i += 1

    # insert with a trailing newline if needed
    ins = import_stmt.rstrip() + "\n"
    return lines[:i] + [ins] + lines[i:]


# -------------------------
# Patch rules (Python-focused)
# -------------------------

RuleFn = Callable[[List[str], List[int], PatchReport], List[str]]


def rule_yaml_safe_load(lines: List[str], targets: List[int], rep: PatchReport) -> List[str]:
    """
    Bandit B506: yaml.load without Loader -> use yaml.safe_load
    """
    changed = False
    for ln in targets:
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        s = lines[idx]
        # very conservative: replace "yaml.load(" with "yaml.safe_load("
        if "yaml.load(" in s and "safe_load" not in s:
            before = s
            after = s.replace("yaml.load(", "yaml.safe_load(")
            lines[idx] = after
            rep.edits.append(PatchEdit("yaml_safe_load", ln, before, after))
            rep.touched_lines.append(ln)
            changed = True

    if changed and not _has_import(lines, "yaml"):
        rep.notes.append("yaml_safe_load applied but no import yaml found; you may need to ensure yaml is imported.")
    if changed:
        rep.applied_rules.append("yaml_safe_load")
    return lines


def rule_eval_literal_eval(lines: List[str], targets: List[int], rep: PatchReport) -> List[str]:
    """
    Bandit B307: eval usage -> ast.literal_eval (best-effort)
    """
    changed = False
    for ln in targets:
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        s = lines[idx]
        # naive patterns: eval( ... )
        if re.search(r"\beval\s*\(", s) and "literal_eval" not in s:
            before = s
            after = re.sub(r"\beval\s*\(", "ast.literal_eval(", s)
            lines[idx] = after
            rep.edits.append(PatchEdit("eval_literal_eval", ln, before, after))
            rep.touched_lines.append(ln)
            changed = True

    if changed:
        if not _has_import(lines, "ast"):
            lines = _insert_import(lines, "import ast")
        rep.applied_rules.append("eval_literal_eval")
    return lines


def rule_subprocess_shell_false(lines: List[str], targets: List[int], rep: PatchReport) -> List[str]:
    """
    Bandit B602/B605: subprocess with shell=True -> shell=False + shlex.split(cmd) (best-effort)
    Works for patterns like:
      subprocess.Popen(cmd, shell=True)
      subprocess.run(cmd, shell=True)
    """
    changed = False
    need_shlex = False

    for ln in targets:
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        s = lines[idx]
        if "subprocess." in s and "shell=True" in s:
            before = s
            s2 = s.replace("shell=True", "shell=False")

            # try to wrap first arg with shlex.split if it looks like a variable or string
            # heuristic: subprocess.xxx( ARG , ... )
            m = re.search(r"(subprocess\.\w+\s*\()\s*([^,]+)\s*,", s2)
            if m:
                arg = m.group(2).strip()
                # avoid double-wrapping lists/tuples
                if not (arg.startswith("[") or arg.startswith("(") or "shlex.split" in arg):
                    s2 = s2.replace(m.group(0), f"{m.group(1)}shlex.split({arg}),")
                    need_shlex = True

            lines[idx] = s2
            rep.edits.append(PatchEdit("subprocess_shell_false", ln, before, s2))
            rep.touched_lines.append(ln)
            changed = True

    if changed:
        if need_shlex and not _has_import(lines, "shlex"):
            lines = _insert_import(lines, "import shlex")
        rep.applied_rules.append("subprocess_shell_false")
    return lines


def rule_os_system_to_subprocess_run(lines: List[str], targets: List[int], rep: PatchReport) -> List[str]:
    """
    Bandit B605-ish: os.system(...) -> subprocess.run(shlex.split(...), check=True)
    Very conservative; only triggers when pattern matches on a single line.
    """
    changed = False
    need_shlex = False
    need_subprocess = False

    for ln in targets:
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        s = lines[idx]
        m = re.search(r"\bos\.system\s*\(\s*(.+?)\s*\)\s*", s)
        if m:
            before = s
            cmd = m.group(1).strip()
            need_subprocess = True
            if not (cmd.startswith("[") or cmd.startswith("(") or "shlex.split" in cmd):
                cmd_expr = f"shlex.split({cmd})"
                need_shlex = True
            else:
                cmd_expr = cmd
            after = re.sub(r"\bos\.system\s*\(\s*(.+?)\s*\)\s*",
                           f"subprocess.run({cmd_expr}, check=True)",
                           s)
            lines[idx] = after
            rep.edits.append(PatchEdit("os_system_to_subprocess_run", ln, before, after))
            rep.touched_lines.append(ln)
            changed = True

    if changed:
        if need_subprocess and not _has_import(lines, "subprocess"):
            lines = _insert_import(lines, "import subprocess")
        if need_shlex and not _has_import(lines, "shlex"):
            lines = _insert_import(lines, "import shlex")
        rep.applied_rules.append("os_system_to_subprocess_run")
    return lines


def rule_requests_verify_true(lines: List[str], targets: List[int], rep: PatchReport) -> List[str]:
    """
    Bandit B501-ish: requests(..., verify=False) -> verify=True
    """
    changed = False
    for ln in targets:
        idx = ln - 1
        if idx < 0 or idx >= len(lines):
            continue
        s = lines[idx]
        if "requests." in s and "verify=False" in s:
            before = s
            after = s.replace("verify=False", "verify=True")
            lines[idx] = after
            rep.edits.append(PatchEdit("requests_verify_true", ln, before, after))
            rep.touched_lines.append(ln)
            changed = True
    if changed:
        rep.applied_rules.append("requests_verify_true")
    return lines


# Map tags -> rules (best-effort)
TAG_RULES: Dict[str, List[str]] = {
    "B506": ["yaml_safe_load"],
    "B307": ["eval_literal_eval"],
    "B602": ["subprocess_shell_false"],
    "B605": ["subprocess_shell_false", "os_system_to_subprocess_run"],
    "B501": ["requests_verify_true"],
}

RULE_IMPL: Dict[str, RuleFn] = {
    "yaml_safe_load": rule_yaml_safe_load,
    "eval_literal_eval": rule_eval_literal_eval,
    "subprocess_shell_false": rule_subprocess_shell_false,
    "os_system_to_subprocess_run": rule_os_system_to_subprocess_run,
    "requests_verify_true": rule_requests_verify_true,
}


def _select_rules_from_tags(tags: List[str]) -> List[str]:
    """
    Choose rule names from tags. If tags are unknown, fall back to a conservative default set.
    """
    chosen: List[str] = []
    for t in tags:
        if t in TAG_RULES:
            chosen.extend(TAG_RULES[t])

    # de-dup keeping order
    seen = set()
    out = []
    for r in chosen:
        if r not in seen:
            out.append(r)
            seen.add(r)

    # fallback: if no known tags, apply a tiny safe subset (won't over-edit)
    if not out:
        out = ["yaml_safe_load", "eval_literal_eval", "subprocess_shell_false", "requests_verify_true"]
    return out


# -------------------------
# Public API
# -------------------------

def graph2code_stub(
    orig_code: str,
    x0_graph: Any,
    x1_graph: Optional[Any] = None,
    *,
    context_window: int = 2,
    max_lines: int = 80,
) -> Tuple[str, PatchReport]:
    """
    Main entry for Step D2 stub.

    Returns:
      patched_code (str)
      report (PatchReport)
    """
    rep = PatchReport()

    lines = orig_code.splitlines(keepends=True)

    target_lines = _get_lines_to_patch(
        x0_graph,
        x1_graph,
        context_window=context_window,
        max_lines=max_lines,
    )
    if not target_lines:
        rep.notes.append("No focus/edit lines found from graph; no changes applied.")
        return orig_code, rep

    # tags
    tags = _extract_tags(x0_graph) or (_extract_tags(x1_graph) if x1_graph is not None else [])
    rep.applied_tags = tags

    # pick rules
    rules = _select_rules_from_tags(tags)
    rep.applied_rules = []  # filled by each rule if it actually changes anything

    # apply rules
    for r in rules:
        fn = RULE_IMPL.get(r, None)
        if fn is None:
            rep.notes.append(f"Unknown rule `{r}` skipped.")
            continue
        lines = fn(lines, target_lines, rep)

    patched = "".join(lines)

    # de-dup touched_lines
    rep.touched_lines = sorted(set(rep.touched_lines))
    # de-dup applied_rules preserving order
    seen = set()
    ar = []
    for r in rep.applied_rules:
        if r not in seen:
            ar.append(r)
            seen.add(r)
    rep.applied_rules = ar

    if not rep.edits:
        rep.notes.append("Rules ran but no matching patterns were found in target region; code unchanged.")

    return patched, rep

