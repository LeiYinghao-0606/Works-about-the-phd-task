import ast
import networkx as nx

class SimpleCFG:
    def __init__(self):
        self.G = nx.DiGraph()
        self.next_id = 0
        self.exit_id = self.new_node("EXIT")

    def new_node(self, label):
        nid = self.next_id
        self.next_id += 1
        self.G.add_node(nid, label=label)
        return nid

    def build(self, stmts, entry_id):
        """
        返回：stmts 这段语句构建后的“最后一个节点集合”（可能有多个出口）
        """
        exits = {entry_id}
        for s in stmts:
            exits = self.add_stmt(s, exits)
        return exits

    def add_stmt(self, s, prev_exits):
        # 为当前语句创建一个节点
        cur = self.new_node(type(s).__name__)
        for p in prev_exits:
            self.G.add_edge(p, cur)

        # return：直接连到 EXIT，并终止后续路径
        if isinstance(s, ast.Return):
            self.G.add_edge(cur, self.exit_id)
            return set()  # 后续不可达

        # if：分 true/false 两条
        if isinstance(s, ast.If):
            # true 分支
            true_entry = self.new_node("IF_TRUE")
            self.G.add_edge(cur, true_entry)
            true_exits = self.build(s.body, true_entry) or {true_entry}

            # false 分支（没有 else 也要有一条路径）
            false_entry = self.new_node("IF_FALSE")
            self.G.add_edge(cur, false_entry)
            false_exits = self.build(s.orelse, false_entry) or {false_entry}

            # 汇合点
            join = self.new_node("IF_JOIN")
            for e in true_exits | false_exits:
                self.G.add_edge(e, join)
            return {join}

        # while：cond -> body（true），cond -> after（false），body_end -> cond（回边）
        if isinstance(s, ast.While):
            cond = cur
            body_entry = self.new_node("WHILE_BODY")
            after_loop = self.new_node("WHILE_AFTER")

            self.G.add_edge(cond, body_entry)      # true
            self.G.add_edge(cond, after_loop)      # false

            body_exits = self.build(s.body, body_entry) or {body_entry}
            for e in body_exits:
                self.G.add_edge(e, cond)           # 回边

            return {after_loop}

        # 其他语句：顺序往下走
        return {cur}

def generate_cfg_from_code(file_path):
    code = open(file_path, "r", encoding="utf-8").read()
    tree = ast.parse(code)

    cfg = SimpleCFG()
    entry = cfg.new_node("ENTRY")
    cfg.build(tree.body, entry)
    return cfg.G
