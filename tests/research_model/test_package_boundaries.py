"""Enforce the one-way research-to-production package boundary."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research_model"


def test_research_model_never_imports_production_src():
    violations = []
    for path in sorted(RESEARCH.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name == "src" or name.startswith("src."):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert not violations, "research_model must not import src:\\n" + "\\n".join(violations)
