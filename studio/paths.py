"""Resolve Studio code/resources separately from shared repository runtime data."""
from pathlib import Path
import os

STUDIO = Path(__file__).resolve().parent
ROOT = STUDIO.parent
MODELS = Path(os.environ.get("STUDIO_MODELS_DIR") or STUDIO / "models").expanduser().resolve()


def voice_directory(*, env=None, studio=STUDIO, root=ROOT):
    """Select an explicit voice bank or Studio-owned resources, then the legacy root."""
    env = os.environ if env is None else env
    bundled = studio / "resources/voice"
    return Path(env.get("STUDIO_VOICE_DIR") or
                (bundled if bundled.is_dir() else root / "voice")).expanduser().resolve()


def pet_directory(*, env=None, studio=STUDIO, root=ROOT):
    """Select an explicit pet source or Studio-owned resources, then the legacy root."""
    env = os.environ if env is None else env
    bundled = studio / "pet"
    return Path(env.get("STUDIO_PET_DIR") or
                (bundled if bundled.is_dir() else root / "pet")).expanduser().resolve()


VOICE = voice_directory()
PET = pet_directory()
