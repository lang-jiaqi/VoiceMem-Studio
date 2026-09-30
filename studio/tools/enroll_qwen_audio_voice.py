"""Create or check the Qwen-Audio-TTS voice used by Studio."""
from __future__ import annotations

import argparse

from studio.core.utils.environment.component import load_environment
from studio.core.utils.tts.qwen_audio_enrollment import ensure_voice


def main(argv=None):
    """Enroll the repository reference after its GitHub URL becomes available."""
    load_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None,
                        help="Public HTTPS recording URL (default: environment or GitHub main)")
    parser.add_argument("--prefix", default="noctelle")
    parser.add_argument("--force", action="store_true", help="Create a new voice ID")
    args = parser.parse_args(argv)
    voice = ensure_voice(url=args.url, prefix=args.prefix, force=args.force)
    print(f"Qwen TTS 音色已就绪：{voice}")


if __name__ == "__main__":
    main()
