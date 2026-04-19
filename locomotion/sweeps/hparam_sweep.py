#!/usr/bin/env python3
"""
Run many training experiments and compare the results.

This script is the experiment manager for hyperparameter sweeps. In plain
language, it does one of two jobs:

1. parent mode:
   read a list of trial settings, launch one training job per trial, and rank
   the finished runs

2. child mode:
   run one single trial with one single set of overrides and write the result
   back to disk

The parent/child split keeps runs isolated from each other so one bad trial
does not poison the whole sweep process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
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
TASK_CHOICES = ("locomotion", "dance")


@dataclass(slots=True)
class TrialRecord:
    """A plain data record describing one finished trial."""

    trial_id: int
    trial_dir: str
    overrides: dict[str, Any]
    success: bool
    exit_code: int
    metric: str
    score: float | None
    summary: dict[str, Any]
    error: str | None


@dataclass(slots=True)
class RunningTrial:
    """Small bookkeeping object for a trial that is still running right now."""

    trial_id: int
    trial_dir: Path
    overrides: dict[str, Any]
    gpu_slot: str
    process: subprocess.Popen[Any]


def _resolve_from_locomotion(path_str: str | os.PathLike[str]) -> Path:
    """Interpret a path relative to `locomotion/` unless it is already absolute."""
    path = Path(path_str)
    return path if path.is_absolute() else (LOCOMOTION_DIR / path).resolve()


def _load_trials(trials_json: Path) -> list[dict[str, Any]]:
    """Read the trial list from JSON and make sure it has the expected shape."""
    data = json.loads(trials_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(
            "Trials JSON must be a list of override objects, for example: "
            '[{"batch_size": 256}, {"batch_size": 512}]'
        )
    return data


def _safe_name(value: str) -> str:
    """Clean up text so it is safe to use inside a folder name."""
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
    """Turn one trial's override settings into a readable folder-name suffix."""
    override_pairs = [f"{key}={overrides[key]}" for key in sorted(overrides)]
    return _safe_name("__".join(override_pairs))


def _read_trial_result(trial_dir: Path) -> dict[str, Any]:
    """Load the result file for one trial, or return a failure stub if it is missing."""
    result_path = trial_dir / RESULT_FILENAME
    if not result_path.exists():
        return {"success": False, "error": f"{RESULT_FILENAME} missing", "summary": {}}
    return json.loads(result_path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    """Write one JSON file, creating parent folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON row to a `.jsonl` log file."""
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload) + "\n")


def _build_sweep_output_dir(args: argparse.Namespace) -> Path:
    """Decide which top-level folder should hold this sweep's output."""
    if args.base_output_dir:
        return _resolve_from_locomotion(args.base_output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (LOCOMOTION_DIR / "outputs" / "sweeps" / f"sweep_{timestamp}").resolve()


def _build_trial_command(
    *,
    trial_id: int,
    xml_path: str,
    task: str,
    trial_dir: Path,
    test_mode: bool,
    overrides: dict[str, Any],
) -> list[str]:
    """Assemble the shell command used to launch one child trial."""
    command = [
        sys.executable,
        str(THIS_FILE),
        "--_run_one_trial",
        "--trial_id",
        str(trial_id),
        "--xml_path",
        xml_path,
        "--task",
        task,
        "--trial_dir",
        str(trial_dir),
        "--overrides_json_inline",
        json.dumps(overrides),
    ]
    if test_mode:
        command.append("--test")

    return command


def _build_child_env(cuda_visible_devices: str | None) -> dict[str, str]:
    """Build the environment variables for one child process."""
    child_env = os.environ.copy()
    if cuda_visible_devices is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return child_env


def _start_trial_process(
    *,
    trial_id: int,
    xml_path: str,
    task: str,
    trial_dir: Path,
    test_mode: bool,
    overrides: dict[str, Any],
    cuda_visible_devices: str | None = None,
) -> subprocess.Popen[Any]:
    """Launch one trial process and immediately return control to the caller."""
    command = _build_trial_command(
        trial_id=trial_id,
        xml_path=xml_path,
        task=task,
        trial_dir=trial_dir,
        test_mode=test_mode,
        overrides=overrides,
    )
    return subprocess.Popen(
        command,
        env=_build_child_env(cuda_visible_devices),
    )


def _parse_gpu_csv(gpu_csv: str | None) -> list[str]:
    """Turn a comma-separated GPU string like `0,1,2` into a clean list."""
    if gpu_csv is None:
        return []
    return [gpu_id.strip() for gpu_id in gpu_csv.split(",") if gpu_id.strip()]


def _discover_gpu_slots() -> list[str]:
    """
    Figure out which GPU ids are available to use.

    First respect any explicit GPU restriction already in the environment. If
    there is none, ask `nvidia-smi` what GPUs exist on the machine.
    """
    visible_devices = _parse_gpu_csv(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices:
        return visible_devices

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _resolve_gpu_slots(args: argparse.Namespace) -> list[str]:
    """Choose the GPU ids that child trials are allowed to use."""
    if args.available_gpus:
        return _parse_gpu_csv(args.available_gpus)
    return _discover_gpu_slots()


def _rank_successful_trials(
    rows: list[TrialRecord],
) -> list[TrialRecord]:
    """Keep only successful trials and sort them from best score to worst."""
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
    task: str,
    max_concurrent_trials: int,
    gpu_slots: list[str],
    num_trials: int,
) -> None:
    """Print the "about to start" summary block for the sweep."""
    print(f"Sweep starting. Trials: {num_trials}")
    print(f"Trials file: {trials_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Metric:      {metric}")
    print(f"Mode:        {'TEST' if test_mode else 'FULL'}")
    print(f"Task:        {task}")
    print(f"Concurrency: {max_concurrent_trials}")
    if gpu_slots:
        print(f"GPU slots:   {', '.join(gpu_slots)}")
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
    """Convert one child's raw result JSON into a clean typed record."""
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


def _prepare_trial_dir(
    trial_dir: Path,
    overrides: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Create the folder for one trial and save its chosen settings."""
    if trial_dir.exists() and overwrite:
        shutil.rmtree(trial_dir)

    trial_dir.mkdir(parents=True, exist_ok=True)
    _write_json(trial_dir / "overrides.json", overrides)


def _record_trial_completion(
    *,
    trial_id: int,
    trial_dir: Path,
    overrides: dict[str, Any],
    metric: str,
    exit_code: int,
    results_jsonl: Path,
    all_rows: list[TrialRecord],
) -> TrialRecord:
    """Record a finished trial in the sweep-wide outputs and print its status."""
    result = _read_trial_result(trial_dir)
    row = _make_trial_record(
        trial_id=trial_id,
        trial_dir=trial_dir,
        overrides=overrides,
        metric=metric,
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
    return row


def sweep_main(args: argparse.Namespace) -> int:
    """
    Run the whole sweep from start to finish.

    This is the parent-mode workflow:

    - load the trial list
    - launch child runs
    - watch them finish
    - build the leaderboard at the end
    """
    if args.max_concurrent_trials < 1:
        print("ERROR: --max_concurrent_trials must be at least 1.")
        return 2

    trials_path = _resolve_from_locomotion(args.trials_json)
    if not trials_path.exists():
        print(f"ERROR: trials_json not found: {trials_path}")
        return 2

    # Read the trial list and optionally trim it for quick experiments.
    trials = _load_trials(trials_path)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    # Prepare the sweep-wide output files and any GPU scheduling information.
    output_dir = _build_sweep_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_jsonl = output_dir / RESULTS_JSONL_FILENAME
    leaderboard_json = output_dir / LEADERBOARD_FILENAME
    best_json = output_dir / BEST_TRIAL_FILENAME
    gpu_slots = _resolve_gpu_slots(args) if args.max_concurrent_trials > 1 else []
    max_concurrent_trials = args.max_concurrent_trials

    if args.max_concurrent_trials > 1:
        if not gpu_slots:
            print(
                "ERROR: concurrent trial mode requires --available_gpus or "
                "visible GPUs from CUDA_VISIBLE_DEVICES/nvidia-smi."
            )
            return 2

        if max_concurrent_trials > len(gpu_slots):
            print(
                "WARNING: requested "
                f"{max_concurrent_trials} concurrent trials but only "
                f"{len(gpu_slots)} GPU slots are available; limiting concurrency."
            )
            max_concurrent_trials = len(gpu_slots)

    _print_sweep_header(
        trials_path=trials_path,
        output_dir=output_dir,
        metric=args.metric,
        test_mode=args.test,
        task=args.task,
        max_concurrent_trials=max_concurrent_trials,
        gpu_slots=gpu_slots,
        num_trials=len(trials),
    )

    all_rows: list[TrialRecord] = []

    if max_concurrent_trials <= 1:
        # Simple mode: run one trial, wait for it, then move to the next.
        for trial_id, overrides in enumerate(trials, start=1):
            tag = _trial_tag(overrides)
            trial_dir = output_dir / f"trial_{trial_id:03d}__{tag}"
            _prepare_trial_dir(trial_dir, overrides, overwrite=args.overwrite)

            print(f"[{trial_id}/{len(trials)}] Running trial: {trial_dir.name}")
            process = _start_trial_process(
                trial_id=trial_id,
                xml_path=args.xml_path,
                task=args.task,
                trial_dir=trial_dir,
                test_mode=args.test,
                overrides=overrides,
            )
            exit_code = process.wait()
            _record_trial_completion(
                trial_id=trial_id,
                trial_dir=trial_dir,
                overrides=overrides,
                metric=args.metric,
                exit_code=exit_code,
                results_jsonl=results_jsonl,
                all_rows=all_rows,
            )
    else:
        # Parallel mode: treat GPUs like worker slots and keep feeding them
        # trials until everything has completed.
        free_gpu_slots = gpu_slots.copy()
        slot_order = {gpu_slot: index for index, gpu_slot in enumerate(gpu_slots)}
        running_trials: list[RunningTrial] = []
        next_trial_index = 0

        while next_trial_index < len(trials) or running_trials:
            while (
                next_trial_index < len(trials)
                and free_gpu_slots
                and len(running_trials) < max_concurrent_trials
            ):
                trial_id = next_trial_index + 1
                overrides = trials[next_trial_index]
                next_trial_index += 1

                tag = _trial_tag(overrides)
                trial_dir = output_dir / f"trial_{trial_id:03d}__{tag}"
                _prepare_trial_dir(trial_dir, overrides, overwrite=args.overwrite)

                gpu_slot = free_gpu_slots.pop(0)
                print(
                    f"[launch {trial_id}/{len(trials)}] "
                    f"GPU {gpu_slot}: {trial_dir.name}"
                )
                process = _start_trial_process(
                    trial_id=trial_id,
                    xml_path=args.xml_path,
                    task=args.task,
                    trial_dir=trial_dir,
                    test_mode=args.test,
                    overrides=overrides,
                    cuda_visible_devices=gpu_slot,
                )
                running_trials.append(
                    RunningTrial(
                        trial_id=trial_id,
                        trial_dir=trial_dir,
                        overrides=overrides,
                        gpu_slot=gpu_slot,
                        process=process,
                    )
                )

            # Poll every running child to see whether any of them finished.
            completed_this_tick = False
            for running_trial in list(running_trials):
                exit_code = running_trial.process.poll()
                if exit_code is None:
                    continue

                completed_this_tick = True
                running_trials.remove(running_trial)
                free_gpu_slots.append(running_trial.gpu_slot)
                free_gpu_slots.sort(key=lambda gpu_slot: slot_order[gpu_slot])

                print(
                    f"[finish {running_trial.trial_id}/{len(trials)}] "
                    f"GPU {running_trial.gpu_slot}: {running_trial.trial_dir.name}"
                )
                _record_trial_completion(
                    trial_id=running_trial.trial_id,
                    trial_dir=running_trial.trial_dir,
                    overrides=running_trial.overrides,
                    metric=args.metric,
                    exit_code=exit_code,
                    results_jsonl=results_jsonl,
                    all_rows=all_rows,
                )

            # If nothing finished this pass, sleep briefly before polling again.
            if running_trials and not completed_this_tick:
                time.sleep(1.0)

    # Once all trials are done, write the final leaderboard and the "best trial" file.
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
    """Pick a safe MuJoCo rendering backend for one child trial process."""
    if "MUJOCO_GL" in os.environ:
        return

    system = platform.system()
    if system == "Linux":
        os.environ["MUJOCO_GL"] = "egl"
    elif system == "Darwin":
        os.environ["MUJOCO_GL"] = "glfw"


def _load_inline_overrides(payload: str) -> dict[str, Any]:
    """Decode the one-trial override JSON passed in by the parent process."""
    overrides = json.loads(payload)
    if not isinstance(overrides, dict):
        raise ValueError("overrides_json_inline must decode to a dict")
    return overrides


def _apply_overrides(config: Any, overrides: dict[str, Any]) -> None:
    """Copy the chosen trial overrides onto the training config, with validation."""
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"Override '{key}' is not a field on TrainingConfig")
        setattr(config, key, value)


def _warn_on_awkward_batch_shapes(config: Any, logger: Any) -> None:
    """
    Warn about setting combinations that are known to be awkward or invalid.

    These warnings try to catch "this probably will not divide evenly" mistakes
    before the expensive part of training burns time.
    """
    if (config.batch_size * config.num_minibatches) % config.num_envs != 0:
        logger.warning(
            "batch_size * num_minibatches is not divisible by num_envs. "
            "Brax PPO requires this relationship."
        )
    if config.batch_size % config.unroll_length != 0:
        logger.warning(
            "batch_size is not divisible by unroll_length. "
            "This may error or reduce efficiency."
        )

    visible_devices = _parse_gpu_csv(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if len(visible_devices) > 1 and config.num_envs % len(visible_devices) != 0:
        logger.warning(
            "num_envs (%s) is not divisible by the visible device count (%s). "
            "Multi-GPU PPO will fail unless these divide evenly.",
            config.num_envs,
            len(visible_devices),
        )


def run_one_trial_main(args: argparse.Namespace) -> int:
    """
    Run one single trial in child mode.

    This is the worker-side workflow: build the config, launch training, and
    write one `trial_result.json` file for the parent to read later.
    """
    _configure_child_mujoco_backend()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Read the exact override values that define this one trial.
    overrides = _load_inline_overrides(args.overrides_json_inline)

    from locomotion.train import train_bittle
    from locomotion.training_config import TrainingConfig
    from locomotion.training_helpers import setup_logging

    logger = setup_logging(trial_dir, level=getattr(__import__("logging"), "INFO"))

    # Start from the task preset, then replace any fields this trial wants to vary.
    config = TrainingConfig.for_task(args.task, test_mode=args.test)
    _apply_overrides(config, overrides)
    _warn_on_awkward_batch_shapes(config, logger)

    # Run training and convert the result into the sweep's standard result format.
    outcome = train_bittle(
        config=config,
        xml_path=args.xml_path,
        output_dir=trial_dir,
        logger=logger,
        task_name=args.task,
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
    """Define the command-line options used by both parent and child modes."""
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
        "--task",
        type=str,
        choices=TASK_CHOICES,
        default="locomotion",
        help="Task/environment to train in each trial.",
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
    parser.add_argument(
        "--max_concurrent_trials",
        type=int,
        default=1,
        help="How many child trials to run at the same time.",
    )
    parser.add_argument(
        "--available_gpus",
        type=str,
        default=None,
        help=(
            "Comma-separated GPU ids reserved for concurrent child trials, "
            "for example '0,1,2,3'."
        ),
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
    """Read terminal arguments and decide whether this process is parent or child."""
    args = build_argparser().parse_args()

    if args._run_one_trial:
        return run_one_trial_main(args)

    if not args.trials_json:
        print("ERROR: --trials_json is required for sweep mode.")
        return 2

    return sweep_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
