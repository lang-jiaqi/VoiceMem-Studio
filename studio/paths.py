"""Repository data roots; Studio model weights have a dedicated directory."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("STUDIO_MODELS_DIR") or ROOT / "studio/models").expanduser().resolve()
