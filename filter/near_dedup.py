"""Near-dedup via normalized AST comparison.

Exact dedup misses the common case: two solutions that are structurally
identical but differ in variable names, whitespace, or comments. This
module normalizes the AST (canonical names, no comments/docstrings by
construction) and dedups on the normalized structure.

Falls back to exact code-hash dedup for code that does not parse.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory
from filter.quality import FilterResult


class _Normalizer(ast.NodeTransformer):
    """Replace every user-chosen name with a canonical placeholder."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="VAR", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "ARG"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = "FN"
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node.name = "FN"
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = "CLS"
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        node.attr = "ATTR"
        return node


def normalize_ast(code: str) -> str | None:
    """Structural key for code. None when the code does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    normalized = _Normalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized)


def structural_key(traj: Trajectory) -> str:
    """Dedup key: normalized AST, or exact hash when unparseable."""
    key = normalize_ast(traj.final_code)
    if key is None:
        return "exact:" + traj.code_hash()
    return "ast:" + key


def near_dedup(trajectories: list[Trajectory]) -> FilterResult:
    """Drop structural duplicates. Keeps the first occurrence.

    Two trajectories are duplicates when their final code has the same
    normalized AST: same control flow, same operations, possibly
    different names. This catches copy-paste solutions that exact
    hashing misses.
    """
    seen: set[str] = set()
    kept = []
    ast_hits = 0
    for t in trajectories:
        key = structural_key(t)
        if key not in seen:
            seen.add(key)
            kept.append(t)
        elif key.startswith("ast:"):
            ast_hits += 1
    return FilterResult(kept=kept, removed=len(trajectories) - len(kept),
                        name=f"near_dedup({ast_hits} structural)")


def duplicate_groups(trajectories: list[Trajectory]) -> dict[str, list[str]]:
    """Map structural key -> trajectory ids, for groups larger than one."""
    groups: dict[str, list[str]] = {}
    for t in trajectories:
        groups.setdefault(structural_key(t), []).append(t.trajectory_id)
    return {k: v for k, v in groups.items() if len(v) > 1}
