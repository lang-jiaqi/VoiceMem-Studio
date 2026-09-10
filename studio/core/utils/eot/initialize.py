"""Initialize semantic turn detection from verified local weights."""
from .component import EndOfTurn
from studio.paths import MODELS

def create():
    return EndOfTurn(str(MODELS / "eot/smart-turn-v3.2-cpu.onnx"), threshold=0.5)
