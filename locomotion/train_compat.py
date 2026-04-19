#!/usr/bin/env python3
"""
Backward-compatible training entrypoint.

The organized implementation now lives in ``locomotion/training/``. This thin
wrapper keeps older commands and imports working while ``Scripts/train.py`` is
the preferred entrypoint for people running training manually.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.training.run import main, train_bittle

__all__ = ["main", "train_bittle"]


if __name__ == "__main__":
    main()
