#!/usr/bin/env python3
"""
Coordinate hyperparameter sweeps for Bittle PPO training.

This script has two modes:

1. Sweep mode:
   The parent process reads a JSON list of config overrides, creates one output
   directory per trial, launches each trial as a child Python process, and
   ranks the successful runs by a chosen metric.

2. Child-trial mode:
   The spawned child process applies one set of overrides to ``TrainingConfig``,
   runs training once, and writes ``trial_result.json`` back into its trial
   directory.

The separation keeps each training run isolated. A failed trial can exit
cleanly without leaving the parent process in a partially initialized JAX or
MuJoCo state.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


# Resolve repository-relative paths once at import time so the script behaves
# the same whether it is launched from the repo root or from locomotion/.
THIS_FILE = Path(__file__).resolve()
SWEEPS_DIR = THIS_FILE.parent
LOCOMOTION_DIR = SWEEPS_DIR.parent
REPO_ROOT = LOCOMOTION_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from locomotion.paths import DEFAULT_SCENE_PATH


RESULT_FILENAME = "trial_result.json"
RESULTS_JSONL_FILENAME = "results.jsonl"
LEADERBOARD_FILENAME = "leaderboard.json"
BEST_TRIAL_FILENAME = "best_trial.json"


@dataclass(slots=True)
class TrialRecord:
    """Serializable summary of one trial within a sweep."""

    trial_id: int
    trial_dir: str
    overrides: dict[str, Any]
    success: bool
    exit_code: int
    metric: str
    score: float | None
    summary: dict[str, Any]
    error: str | None


def _resolve_from_locomotion(path_str: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to ``locomotion/`` unless it is absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else (LOCOMOTION_DIR / path).resolve()


def _load_trials(trials_json: Path) -> list[dict[str, Any]]:
    """Load and validate the sweep override list."""
    data = json.loads(trials_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(
            "Trials JSON must be a list of override objects, for example: "
            '[{"batch_size": 256}, {"batch_size": 512}]'
        )
    return data


def _safe_name(value: str) -> str:
    """Convert a human-readable override string into a filesystem-safe token."""
    translation_table = str.maketrans(
        {
            " ": "",
            "/": "_",
            "\\": "_",
            ":": "_",
            "|": "_",
            '"': "",
            "'": "",
        }
    )
    return value.translate(translation_table)


def _trial_tag(overrides: dict[str, Any]) -> str:
    """Build a stable directory suffix from sorted override key/value pairs."""
    override_pairs = [f"{key}={overrides[key]}" for key in sorted(overrides)]
    return _safe_name("__".join(override_pairs))


def _read_trial_result(trial_dir: Path) -> dict[str, Any]:
    """Read the trial result written by the child process, if present."""
    result_path = trial_dir / RESULT_FILENAME
    if not result_path.exists():
        return {"success": False, "error": f"{RESULT_FILENAME} missing", "summary": {}}
    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON with parent directories created automatically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON object per line for easy streaming and inspection."""
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload) + "\n")


def _build_sweep_output_dir(args: argparse.Namespace) -> Path:
    """Choose the sweep output directory from CLI input or a timestamp."""
    if args.base_output_dir:
        return _resolve_from_locomotion(args.base_output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (LOCOMOTION_DIR / "outputs" / "sweeps" / f"sweep_{timestamp}").resolve()


def _spawn_trial_process(
    *,
    trial_id: int,
    xml_path: str,
    trial_dir: Path,
    test_mode: bool,
    overrides: dict[str, Any],
) -> int:
    """Launch a child process that runs exactly one training trial."""
    command = [
        sys.executable,
        str(THIS_FILE),
        "--_run_one_trial",
        "--trial_id",
        str(trial_id),
        "--xml_path",
        xml_path,
        "--trial_dir",
        str(trial_dir),
        "--overrides_json_inline",
        json.dumps(overrides),
    ]
    if test_mode:
        command.append("--test")

    return subprocess.run(command, check=False).returncode


def _rank_successful_trials(
    rows: list[TrialRecord],
) -> list[TrialRecord]:
    """Return successful trials sorted by descending score."""
    scored_rows = [
        row
        for row in rows
        if row.success and isinstance(row.score, (int, float))
    ]
    return sorted(scored_rows, key=lambda row: float(row.score), reverse=True)


def _print_sweep_header(
    *,
    trials_path: Path,
    output_dir: Path,
    metric: str,
    test_mode: bool,
    num_trials: int,
) -> None:
    """Print a compact summary of the sweep about to run."""
    print(f"Sweep starting. Trials: {num_trials}")
    print(f"Trials file: {trials_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Metric:      {metric}")
    print(f"Mode:        {'TEST' if test_mode else 'FULL'}")
    print("")


def _make_trial_record(
    *,
    trial_id: int,
    trial_dir: Path,
    overrides: dict[str, Any],
    metric: str,
    exit_code: int,
    result: dict[str, Any],
) -> TrialRecord:
    """Normalize one child's result into a typed sweep record."""
    summary = result.get("summary", {}) or {}
    success = bool(result.get("success", False))
    score = summary.get(metric)

    if not isinstance(score, (int, float)):
        score = None

    return TrialRecord(
        trial_id=trial_id,
        trial_dir=str(trial_dir),
        overrides=overrides,
        success=success,
        exit_code=exit_code,
        metric=metric,
        score=float(score) if score is not None else None,
        summary=summary,
        error=result.get("error"),
    )


def sweep_main(args: argparse.Namespace) -> int:
    """Run all requested trials and build the final leaderboard."""
    trials_path = _resolve_from_locomotion(args.trials_json)
    if not trials_path.exists():
        print(f"ERROR: trials_json not found: {trials_path}")
        return 2

    trials = _load_trials(trials_path)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    output_dir = _build_sweep_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / RESULTS_JSONL_FILENAME
    leaderboard_json = output_dir / LEADERBOARD_FILENAME
    best_json = output_dir / BEST_TRIAL_FILENAME

    _print_sweep_header(
        trials_path=trials_path,
        output_dir=output_dir,
        metric=args.metric,
        test_mode=args.test,
        num_trials=len(trials),
    )

    all_rows: list[TrialRecord] = []

    for trial_id, overrides in enumerate(trials, start=1):
        tag = _trial_tag(overrides)
        trial_dir = output_dir / f"trial_{trial_id:03d}__{tag}"

        if trial_dir.exists() and args.overwrite:
            shutil.rmtree(trial_dir)

        trial_dir.mkdir(parents=True, exist_ok=True)
        _write_json(trial_dir / "overrides.json", overrides)

        print(f"[{trial_id}/{len(trials)}] Running trial: {trial_dir.name}")
        exit_code = _spawn_trial_process(
            trial_id=trial_id,
            xml_path=args.xml_path,
            trial_dir=trial_dir,
            test_mode=args.test,
            overrides=overrides,
        )

        result = _read_trial_result(trial_dir)
        row = _make_trial_record(
            trial_id=trial_id,
            trial_dir=trial_dir,
            overrides=overrides,
            metric=args.metric,
            exit_code=exit_code,
            result=result,
        )
        all_rows.append(row)
        _append_jsonl(results_jsonl, asdict(row))

        if row.success:
            print(f"  done: score={row.score}")
        else:
            print(f"  failed: {row.error}")
        print("")

    leaderboard = _rank_successful_trials(all_rows)
    _write_json(leaderboard_json, [asdict(row) for row in leaderboard])

    if not leaderboard:
        print("No successful trials to rank.")
        return 1

    best = leaderboard[0]
    _write_json(best_json, asdict(best))

    print("Best trial:")
    print(f"  trial_id: {best.trial_id}")
    print(f"  score:    {best.score}")
    print(f"  dir:      {best.trial_dir}")
    print(f"  overrides:{best.overrides}")
    return 0


def _configure_child_mujoco_backend() -> None:
    """Apply a platform-appropriate default MuJoCo backend for child trials."""
    if "MUJOCO_GL" in os.environ:
        return

    system = platform.system()
    if system == "Linux":
        os.environ["MUJOCO_GL"] = "egl"
    elif system == "Darwin":
        os.environ["MUJOCO_GL"] = "glfw"


def _load_inline_overrides(payload: str) -> dict[str, Any]:
    """Decode the inline JSON blob passed from the parent process."""
    overrides = json.loads(payload)
    if not isinstance(overrides, dict):
        raise ValueError("overrides_json_inline must decode to a dict")
    return overrides


def _apply_overrides(config: Any, overrides: dict[str, Any]) -> None:
    """Apply sweep overrides to ``TrainingConfig`` with strict field checking."""
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Override '{key}' is not a field on TrainingConfig")
        setattr(config, key, value)


def _warn_on_awkward_batch_shapes(config: Any, logger: Any) -> None:
    """Warn about common batch-size relationships that tend to cause issues."""
    if config.batch_size % config.num_minibatches != 0:
        logger.warning(
            "batch_size is not divisible by num_minibatches. "
            "This may error or reduce efficiency."
        )
    if config.batch_size % config.unroll_length != 0:
        logger.warning(
            "batch_size is not divisible by unroll_length. "
            "This may error or reduce efficiency."
        )


def run_one_trial_main(args: argparse.Namespace) -> int:
    """Run one child training job and write ``trial_result.json``."""
    _configure_child_mujoco_backend()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    overrides = _load_inline_overrides(args.overrides_json_inline)

    from locomotion.train import train_bittle
    from locomotion.training_config import TrainingConfig
    from locomotion.training_helpers import setup_logging

    logger = setup_logging(trial_dir, level=getattr(__import__("logging"), "INFO"))

    config = TrainingConfig(test_mode=args.test)
    _apply_overrides(config, overrides)
    _warn_on_awkward_batch_shapes(config, logger)

    outcome = train_bittle(
        config=config,
        xml_path=args.xml_path,
        output_dir=trial_dir,
        logger=logger,
    )

    result = {
        "success": bool(outcome.get("success", False)),
        "overrides": overrides,
        "trial_dir": str(trial_dir),
        "summary": outcome.get("summary", {}) if outcome.get("success") else {},
        "error": outcome.get("error"),
    }
    _write_json(trial_dir / RESULT_FILENAME, result)
    return 0 if result["success"] else 1


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI shared by both the parent and child modes."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a Bittle PPO hyperparameter sweep or, internally, a single "
            "child trial spawned by the sweep coordinator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default=str(DEFAULT_SCENE_PATH),
        help="Path to the MuJoCo scene XML used for training.",
    )
    parser.add_argument(
        "--trials_json",
        type=str,
        help="Path to a JSON list of override dictionaries.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default=None,
        help="Sweep output directory, relative to locomotion/ when not absolute.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="best_reward",
        choices=["best_reward", "final_reward", "mean_reward"],
        help="Summary metric used to rank successful trials.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run each trial in test mode instead of full training mode.",
    )
    parser.add_argument(
        "--max_trials",
        type=int,
        default=None,
        help="Limit the number of trials read from the input JSON.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing trial directory before rerunning that trial.",
    )

    # Internal child-process arguments. These are intentionally hidden from the
    # main sweep workflow but remain stable so the parent can spawn children.
    parser.add_argument("--_run_one_trial", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--trial_id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--trial_dir", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--overrides_json_inline",
        type=str,
        default="{}",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    """Parse CLI arguments and dispatch to parent or child mode."""
    args = build_argparser().parse_args()

    if args._run_one_trial:
        return run_one_trial_main(args)

    if not args.trials_json:
        print("ERROR: --trials_json is required for sweep mode.")
        return 2

    return sweep_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
