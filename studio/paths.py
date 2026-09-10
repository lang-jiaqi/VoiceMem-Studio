"""Repository data roots; Studio model weights have a dedicated directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "studio/models"
