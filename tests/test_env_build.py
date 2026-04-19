"""Smoke-test that the Bittle environment can be constructed from repo root."""

from __future__ import annotations

import sys
from pathlib import Path

from brax import envs

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.tasks.bittle_walk_env import BittleWalkingEnv
from locomotion.paths import DEFAULT_SCENE_PATH


def main() -> int:
    env_name = "bittle_walking"
    xml_path = str(DEFAULT_SCENE_PATH)

    envs.register_environment(env_name, BittleWalkingEnv)

    try:
        envs.get_environment(env_name, xml_path=xml_path)
        print(f"Successfully built environment {env_name} from {xml_path}")
        return 0
    except Exception as exc:
        print(f"Failed to build environment {env_name} from {xml_path}")
        print(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
