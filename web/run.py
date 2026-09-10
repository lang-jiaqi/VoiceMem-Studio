"""Compatibility launcher. Prefer python -m studio."""
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from studio.core.core import main
    main()
