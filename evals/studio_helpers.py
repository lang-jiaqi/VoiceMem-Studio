"""Load actual Studio methods against isolated state without starting models."""
import ast
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "studio/core/utils"


def studio_tree():
    """Collect utility methods and contracts for focused AST regressions."""
    body = []
    for path in sorted(UTILS.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and path.parent.name != "contracts":
                body.extend(node.body)
            else:
                body.append(node)
    return ast.Module(body=body, type_ignores=[])


class FixtureState:
    """Expose test fixture entries through the same attributes as VoiceAgent."""
    def __init__(self, namespace):
        object.__setattr__(self, "namespace", namespace)

    def __getattr__(self, name):
        try:
            return self.namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self.namespace[name] = value


def execute(nodes, namespace):
    """Compile unchanged method bodies and bind only their instance argument."""
    state = namespace.setdefault("self", FixtureState(namespace))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "studio fixture", "exec"), namespace)
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.args.args and node.args.args[0].arg == "self":
                namespace[node.name] = MethodType(namespace[node.name], state)
    return namespace


def studio_source():
    return "\n".join(path.read_text() for path in sorted(UTILS.rglob("*.py")))
