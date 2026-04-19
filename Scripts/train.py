#!/usr/bin/env python3
"""
User-facing entrypoint for starting a training run.

This keeps the things a person runs directly under ``Scripts/``, while the
implementation details live under ``locomotion/training/``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.training.run import main


if __name__ == "__main__":
    main()
