# -*- coding: utf-8 -*-
"""
Output:
  - nx.DiGraph where:
      node attrs: ast_type, lineno, end_lineno, code
      edge attrs: etype in {NEXT, T, F, BACK, BREAK, CONTINUE}
  - line_to_nodes: Dict[int, List[node_id]] for mapping focus_lines -> focus_nodes

Scope (practical & sufficient for PromSec-style PoC):
  - Supports: If, For, While, Break, Continue, Return, Raise, Expr/Assign/Call, etc.
  - Treats many other statements as "simple" sequential nodes.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import networkx as nx


EdgeType = str


@dataclass
class BlockResult:
    """
    entry: entry node id of the built block (None if empty)
    exits: nodes that should connect to the "next" after this block (normal fallthrough)
    breaks: nodes that are break statements in this block
    continues: nodes that are continue statements in this block
    terminals: nodes that terminate control flow (return/raise), included for bookkeeping
    """
    entry: Optional[int]
    exits: Set[int]
    breaks: Set[int]
    continues: Set[int]
    terminals: Set[int]


class CFGBuilder:
    def __init__(self, source: str):
        self.source = source
        self.g = nx.DiGraph()
        self._nid = 0
        self.line_to_nodes: Dict[int, List[int]] = {}

    def new_node(self, stmt: ast.AST, ast_type: Optional[str] = None) -> int:
        nid = self._nid
        self._nid += 1

        tname = ast_type or type(stmt).__name__
        lineno = int(getattr(stmt, "lineno", 0) or 0)
        end_lineno = int(getattr(stmt, "end_lineno", lineno) or lineno)

        code = self._get_source_segment(stmt)

        self.g.add_node(
            nid,
            ast_type=tname,
            lineno=lineno if lineno > 0 else None,
            end_lineno=end_lineno if end_lineno > 0 else None,
            code=code,
        )

        # line -> node mapping (use lineno; also map end_lineno for multi-line blocks)
        if lineno and lineno > 0:
            self.line_to_nodes.setdefault(lineno, []).append(nid)
        if end_lineno and end_lineno > 0 and end_lineno != lineno:
            self.line_to_nodes.setdefault(end_lineno, []).append(nid)

        return nid

    def add_edge(self, u: int, v: int, etype: EdgeType) -> None:
        self.g.add_edge(u, v, etype=etype)

    def _get_source_segment(self, stmt: ast.AST) -> str:
        # Prefer ast.get_source_segment for accurate snippet
        try:
            seg = ast.get_source_segment(self.source, stmt)
            if seg:
                return seg.strip()
        except Exception:
            pass

        # Fallback: use the line at lineno
        lineno = int(getattr(stmt, "lineno", 0) or 0)
        if lineno <= 0:
            return type(stmt).__name__
        lines = self.source.splitlines()
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1].rstrip()
        return type(stmt).__name__

    # ---------------------------
    # Public API
    # ---------------------------
    def build_module_cfg(self, tree: ast.AST) -> Tuple[nx.DiGraph, Dict[int, List[int]]]:
        # Build CFG for module-level statements
        if not isinstance(tree, ast.Module):
            raise TypeError("Expected ast.Module")

        res = self._build_block(tree.body, loop_header=None)
        # (Optional) add explicit ENTRY/EXIT
        entry = self._add_entry_exit(res.entry, res.exits)
        return self.g, self.line_to_nodes

    # ---------------------------
    # Core block / statement builders
    # ---------------------------
    def _build_block(self, stmts: Sequence[ast.stmt], loop_header: Optional[int]) -> BlockResult:
        """
        Build CFG for a sequence of statements.
        loop_header: if inside a loop, used to resolve continue edges.
        """
        entry: Optional[int] = None
        exits: Set[int] = set()
        breaks: Set[int] = set()
        continues: Set[int] = set()
        terminals: Set[int] = set()

        prev_exits: Set[int] = set()  # fallthrough nodes to connect to next statement

        for stmt in stmts:
            r = self._build_stmt(stmt, loop_header=loop_header)

            if entry is None:
                entry = r.entry

            # Connect previous fallthroughs to this statement entry
            if r.entry is not None:
                for p in prev_exits:
                    self.add_edge(p, r.entry, "NEXT")

            # Propagate special exits upward
            breaks |= r.breaks
            continues |= r.continues
            terminals |= r.terminals

            # New prev_exits are this statement's normal exits
            prev_exits = set(r.exits)

        exits = prev_exits  # block exits are the last statement's fallthrough exits
        return BlockResult(entry=entry, exits=exits, breaks=breaks, continues=continues, terminals=terminals)

    def _build_stmt(self, stmt: ast.stmt, loop_header: Optional[int]) -> BlockResult:
        """
        Build CFG for a single statement, returning its entry and normal exits.
        """
        # --- If / elif / else ---
        if isinstance(stmt, ast.If):
            return self._build_if(stmt, loop_header=loop_header)

        # --- While ---
        if isinstance(stmt, ast.While):
            return self._build_while(stmt)

        # --- For ---
        if isinstance(stmt, ast.For):
            return self._build_for(stmt)

        # --- Break / Continue ---
        if isinstance(stmt, ast.Break):
            n = self.new_node(stmt)
            return BlockResult(entry=n, exits=set(), breaks={n}, continues=set(), terminals=set())

        if isinstance(stmt, ast.Continue):
            n = self.new_node(stmt)
            # continue will be resolved by loop builder; record it here
            return BlockResult(entry=n, exits=set(), breaks=set(), continues={n}, terminals=set())

        # --- Return / Raise (terminal) ---
        if isinstance(stmt, ast.Return) or isinstance(stmt, ast.Raise):
            n = self.new_node(stmt)
            return BlockResult(entry=n, exits=set(), breaks=set(), continues=set(), terminals={n})

        # --- Try/Except/Finally (simple conservative approximation) ---
        if isinstance(stmt, ast.Try):
            return self._build_try(stmt, loop_header=loop_header)

        # --- FunctionDef / ClassDef: treat as simple nodes at module-level ---
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            n = self.new_node(stmt)
            return BlockResult(entry=n, exits={n}, breaks=set(), continues=set(), terminals=set())

        # --- Default: simple sequential node ---
        n = self.new_node(stmt)
        return BlockResult(entry=n, exits={n}, breaks=set(), continues=set(), terminals=set())

    def _build_if(self, stmt: ast.If, loop_header: Optional[int]) -> BlockResult:
        cond = self.new_node(stmt, ast_type="If")  # represent the branch node at stmt.lineno

        # Build body / orelse blocks
        body_res = self._build_block(stmt.body, loop_header=loop_header) if stmt.body else BlockResult(None, set(), set(), set(), set())
        else_res = self._build_block(stmt.orelse, loop_header=loop_header) if stmt.orelse else BlockResult(None, set(), set(), set(), set())

        exits: Set[int] = set()
        breaks: Set[int] = set()
        continues: Set[int] = set()
        terminals: Set[int] = set()

        # True edge
        if body_res.entry is not None:
            self.add_edge(cond, body_res.entry, "T")
            exits |= body_res.exits
        else:
            # empty true-branch: cond falls through
            exits.add(cond)

        breaks |= body_res.breaks
        continues |= body_res.continues
        terminals |= body_res.terminals

        # False edge
        if else_res.entry is not None:
            self.add_edge(cond, else_res.entry, "F")
            exits |= else_res.exits
        else:
            # no else: false falls through
            exits.add(cond)

        breaks |= else_res.breaks
        continues |= else_res.continues
        terminals |= else_res.terminals

        # The cond node itself should not remain in exits if it has outgoing branch to non-empty,
        # but allowing it in exits is fine because it will be connected by NEXT.
        # We keep it as a conservative fallthrough representative.

        return BlockResult(entry=cond, exits=exits, breaks=breaks, continues=continues, terminals=terminals)

    def _build_while(self, stmt: ast.While) -> BlockResult:
        header = self.new_node(stmt, ast_type="While")

        # body
        body_res = self._build_block(stmt.body, loop_header=header) if stmt.body else BlockResult(None, set(), set(), set(), set())

        # While condition true -> body, false -> exit path
        if body_res.entry is not None:
            self.add_edge(header, body_res.entry, "T")
        else:
            # empty body, still a loop header; true edge returns to header (degenerate)
            self.add_edge(header, header, "BACK")

        # Normal fallthrough from body goes back to header (loop back)
        for x in body_res.exits:
            self.add_edge(x, header, "BACK")

        # continues go back to header
        for c in body_res.continues:
            self.add_edge(c, header, "CONTINUE")

        # breaks exit the loop; we mark as breaks to be linked by caller of loop (here we resolve to loop-exit)
        # but since we know loop exit node set, we can treat them as exits via BREAK edges
        loop_exits: Set[int] = {header}  # false edge falls through from header itself
        for b in body_res.breaks:
            # break jumps to loop exit (represented by header in exits set; caller connects to next stmt)
            self.add_edge(b, header, "BREAK")
            loop_exits.add(b)  # also include as representative, though edge already indicates break

        # While orelse executes if loop ends normally (condition false), not if broken
        # In Python, orelse runs when loop terminates without break.
        # We'll connect header(false) -> orelse entry via "F".
        orelse_res = self._build_block(stmt.orelse, loop_header=None) if stmt.orelse else BlockResult(None, set(), set(), set(), set())
        exits: Set[int] = set()
        breaks: Set[int] = set()
        continues: Set[int] = set()
        terminals: Set[int] = set()

        if orelse_res.entry is not None:
            self.add_edge(header, orelse_res.entry, "F")
            exits |= orelse_res.exits
            breaks |= orelse_res.breaks
            continues |= orelse_res.continues
            terminals |= orelse_res.terminals
        else:
            # no orelse: header false falls through
            exits.add(header)

        # Merge terminals from body (return/raise)
        terminals |= body_res.terminals

        # After loop: also allow fallthrough from header(false) and break nodes.
        # We represent loop exits by including header and break nodes in exits set;
        # caller will connect them to the next statement via NEXT.
        exits |= {header}
        exits |= body_res.breaks  # break nodes are terminal for loop, but not program; connect outwards

        return BlockResult(entry=header, exits=exits, breaks=breaks, continues=continues, terminals=terminals)

    def _build_for(self, stmt: ast.For) -> BlockResult:
        header = self.new_node(stmt, ast_type="For")

        body_res = self._build_block(stmt.body, loop_header=header) if stmt.body else BlockResult(None, set(), set(), set(), set())

        if body_res.entry is not None:
            self.add_edge(header, body_res.entry, "T")
        else:
            # empty body: true edge loops back
            self.add_edge(header, header, "BACK")

        for x in body_res.exits:
            self.add_edge(x, header, "BACK")

        for c in body_res.continues:
            self.add_edge(c, header, "CONTINUE")

        for b in body_res.breaks:
            self.add_edge(b, header, "BREAK")

        # for-else: runs if loop completes without break
        orelse_res = self._build_block(stmt.orelse, loop_header=None) if stmt.orelse else BlockResult(None, set(), set(), set(), set())

        exits: Set[int] = set()
        breaks: Set[int] = set()
        continues: Set[int] = set()
        terminals: Set[int] = set()

        if orelse_res.entry is not None:
            self.add_edge(header, orelse_res.entry, "F")
            exits |= orelse_res.exits
            breaks |= orelse_res.breaks
            continues |= orelse_res.continues
            terminals |= orelse_res.terminals
        else:
            exits.add(header)

        terminals |= body_res.terminals

        exits |= {header}
        exits |= body_res.breaks

        return BlockResult(entry=header, exits=exits, breaks=breaks, continues=continues, terminals=terminals)

    def _build_try(self, stmt: ast.Try, loop_header: Optional[int]) -> BlockResult:
        """
        Conservative approximation:
          - represent Try as a node, connect to try-body entry via NEXT
          - except handlers are connected from Try node as separate NEXT edges
          - finally is appended after normal exits + except exits
        This is enough for PoC-level graph supervision without full exception CFG semantics.
        """
        tn = self.new_node(stmt, ast_type="Try")

        body_res = self._build_block(stmt.body, loop_header=loop_header) if stmt.body else BlockResult(None, set(), set(), set(), set())
        if body_res.entry is not None:
            self.add_edge(tn, body_res.entry, "NEXT")

        handler_entries: List[int] = []
        handler_exits: Set[int] = set()
        handler_breaks: Set[int] = set()
        handler_continues: Set[int] = set()
        handler_terminals: Set[int] = set()

        for h in stmt.handlers:
            hr = self._build_block(h.body, loop_header=loop_header) if h.body else BlockResult(None, set(), set(), set(), set())
            if hr.entry is not None:
                self.add_edge(tn, hr.entry, "NEXT")
                handler_entries.append(hr.entry)
            handler_exits |= hr.exits
            handler_breaks |= hr.breaks
            handler_continues |= hr.continues
            handler_terminals |= hr.terminals

        orelse_res = self._build_block(stmt.orelse, loop_header=loop_header) if stmt.orelse else BlockResult(None, set(), set(), set(), set())

        # Normal flow: try-body exits go to orelse (if any) else fall through
        exits: Set[int] = set()
        if orelse_res.entry is not None:
            for x in body_res.exits:
                self.add_edge(x, orelse_res.entry, "NEXT")
            exits |= orelse_res.exits
        else:
            exits |= body_res.exits

        # Exception flow: handler exits also fall through
        exits |= handler_exits

        breaks = body_res.breaks | handler_breaks | orelse_res.breaks
        continues = body_res.continues | handler_continues | orelse_res.continues
        terminals = body_res.terminals | handler_terminals | orelse_res.terminals

        # Finally: appended after all normal/except/orelse exits
        final_res = self._build_block(stmt.finalbody, loop_header=loop_header) if stmt.finalbody else BlockResult(None, set(), set(), set(), set())
        if final_res.entry is not None:
            # connect every current exit to finally
            for x in list(exits) if exits else [tn]:
                self.add_edge(x, final_res.entry, "NEXT")
            exits = final_res.exits
            breaks |= final_res.breaks
            continues |= final_res.continues
            terminals |= final_res.terminals

        # If try-body is empty and no handlers, allow tn as fallthrough
        if body_res.entry is None and not handler_entries and final_res.entry is None:
            exits.add(tn)

        return BlockResult(entry=tn, exits=exits, breaks=breaks, continues=continues, terminals=terminals)

    def _add_entry_exit(self, entry: Optional[int], exits: Set[int]) -> int:
        """
        Add explicit ENTRY/EXIT nodes for convenience.
        """
        entry_node = self.g.number_of_nodes()
        self.g.add_node(entry_node, ast_type="ENTRY", lineno=None, end_lineno=None, code="ENTRY")

        exit_node = entry_node + 1
        self.g.add_node(exit_node, ast_type="EXIT", lineno=None, end_lineno=None, code="EXIT")

        if entry is not None:
            self.add_edge(entry_node, entry, "NEXT")
        else:
            # empty module
            self.add_edge(entry_node, exit_node, "NEXT")

        # Connect all fallthrough exits to EXIT
        for x in exits:
            self.add_edge(x, exit_node, "NEXT")

        return entry_node


def build_cfg_from_code(code: str) -> Tuple[nx.DiGraph, Dict[int, List[int]]]:
    """
    Convenience wrapper: code string -> CFG graph + line_to_nodes.
    """
    tree = ast.parse(code)
    b = CFGBuilder(code)
    g, line_to_nodes = b.build_module_cfg(tree)
    return g, line_to_nodes


if __name__ == "__main__":
    sample = """
def f(x):
    if x > 0:
        print("pos")
    else:
        print("neg")
    for i in range(3):
        if i == 1:
            continue
        if i == 2:
            break
        print(i)
    return 1
"""
    g, m = build_cfg_from_code(sample)
    print("nodes:", g.number_of_nodes(), "edges:", g.number_of_edges())
    for u, v, d in g.edges(data=True):
        print(u, "->", v, d.get("etype"))
    print("line_to_nodes keys:", sorted(list(m.keys()))[:10])

