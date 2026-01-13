from src.analysis.bandit_runner import run_bandit_on_code
from src.analysis.score import extract_security_signals
from src.graphs.ast_cfg_python import build_cfg_from_code
from src.graphs.pyg_convert import nx_cfg_to_pyg
from src.graphs.subgraph_focus import (
    attach_security_signals,
    focus_lines_to_focus_nodes_pyg,
    extract_focus_subgraph,
)

code_str = open("some.py", "r", encoding="utf-8").read()

# A1 + A2
rep = run_bandit_on_code(code_str)
sig = extract_security_signals(rep, context_window=1, topk_tags=8)

# A3 + A4
cfg, line_to_nodes = build_cfg_from_code(code_str)
data, nx2pyg = nx_cfg_to_pyg(cfg)

# focus_lines -> focus_nodes(pyg)
focus_nodes = focus_lines_to_focus_nodes_pyg(sig.focus_lines, line_to_nodes, nx2pyg)

# attach signals
data = attach_security_signals(
    data,
    risk_score=sig.risk_score,
    risk_score_log=sig.risk_score_log,
    safe=sig.safe,
    cond_tags=sig.cond_tags,
)

# A5: extract local subgraph + edit_mask
sub = extract_focus_subgraph(
    data,
    focus_nodes=focus_nodes,
    num_hops=2,
    edit_hops=1,
    empty_focus_policy="all",
)

