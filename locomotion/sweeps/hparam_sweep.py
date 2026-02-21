#!/usr/bin/env python3
"""
Hyperparameter sweep runner for Bittle PPO training.

Put this file at: locomotion/sweeps/hparam_sweep.py

Run (recommended from locomotion/):
  cd locomotion
  uv run sweeps/hparam_sweep.py --trials_json sweeps/trials_2080ti_screen.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Ensure parent "locomotion/" is importable even when running from sweeps/
_THIS_FILE = Path(__file__).resolve()
SWEEPS_DIR = _THIS_FILE.parent                 # locomotion/sweeps
LOCOMOTION_DIR = SWEEPS_DIR.parent             # locomotion
if str(LOCOMOTION_DIR) not in sys.path:
    sys.path.insert(0, str(LOCOMOTION_DIR))


def _resolve_from_locomotion(p: str) -> Path:
    """Resolve a path relative to locomotion/ unless it is already absolute."""
    pp = Path(p)
    return pp if pp.is_absolute() else (LOCOMOTION_DIR / pp).resolve()


def _load_trials(trials_json: Path) -> List[Dict[str, Any]]:
    data = json.loads(trials_json.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError("Trials JSON must be a list of objects, like: [{...}, {...}]")
    return data


def _safe_name(s: str) -> str:
    return (
        s.replace(" ", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("|", "_")
        .replace('"', "")
        .replace("'", "")
    )


def _trial_tag(overrides: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(overrides.keys()):
        parts.append(f"{k}={overrides[k]}")
    return _safe_name("__".join(parts))


def _read_trial_result(trial_dir: Path) -> Dict[str, Any]:
    p = trial_dir / "trial_result.json"
    if not p.exists():
        return {"success": False, "error": "trial_result.json missing", "summary": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run_child_trial(
    python_exe: str,
    this_file: Path,
    trial_id: int,
    xml_path: str,
    trial_dir: Path,
    test_mode: bool,
    overrides: Dict[str, Any],
) -> int:
    """
    Spawns a child process to run exactly one trial.
    """
    cmd = [
        python_exe,
        str(this_file),
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
        cmd.append("--test")

    return subprocess.call(cmd)


def sweep_main(args: argparse.Namespace) -> int:
    trials_path = _resolve_from_locomotion(args.trials_json)
    if not trials_path.exists():
        print(f"ERROR: trials_json not found: {trials_path}")
        return 2

    trials = _load_trials(trials_path)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.base_output_dir:
        base_out = _resolve_from_locomotion(args.base_output_dir)
    else:
        base_out = (LOCOMOTION_DIR / "outputs" / "sweeps" / f"sweep_{ts}").resolve()

    base_out.mkdir(parents=True, exist_ok=True)

    results_jsonl = base_out / "results.jsonl"
    leaderboard_json = base_out / "leaderboard.json"
    best_json = base_out / "best_trial.json"

    print(f"Sweep starting. Trials: {len(trials)}")
    print(f"Trials file: {trials_path}")
    print(f"Output dir:  {base_out}")
    print(f"Metric:      {args.metric}")
    print(f"Mode:        {'TEST' if args.test else 'FULL'}")
    print("")

    all_rows: List[Dict[str, Any]] = []

    this_file = Path(__file__).resolve()
    python_exe = sys.executable

    for i, overrides in enumerate(trials, start=1):
        tag = _trial_tag(overrides)
        trial_dir = base_out / f"trial_{i:03d}__{tag}"

        if trial_dir.exists() and args.overwrite:
            shutil.rmtree(trial_dir)

        trial_dir.mkdir(parents=True, exist_ok=True)
        _write_json(trial_dir / "overrides.json", overrides)

        print(f"[{i}/{len(trials)}] Running trial: {trial_dir.name}")
        exit_code = _run_child_trial(
            python_exe=python_exe,
            this_file=this_file,
            trial_id=i,
            xml_path=args.xml_path,
            trial_dir=trial_dir,
            test_mode=args.test,
            overrides=overrides,
        )

        result = _read_trial_result(trial_dir)
        summary = result.get("summary", {}) or {}
        success = bool(result.get("success", False))
        score = summary.get(args.metric, None)

        row = {
            "trial_id": i,
            "trial_dir": str(trial_dir),
            "overrides": overrides,
            "success": success,
            "exit_code": exit_code,
            "metric": args.metric,
            "score": score,
            "summary": summary,
            "error": result.get("error"),
        }
        all_rows.append(row)

        with results_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if success:
            print(f"  done: score={score}")
        else:
            print(f"  failed: {row.get('error')}")
        print("")

    scored: List[Tuple[float, Dict[str, Any]]] = []
    for r in all_rows:
        if not r["success"]:
            continue
        s = r.get("score")
        if isinstance(s, (int, float)):
            scored.append((float(s), r))

    scored.sort(key=lambda x: x[0], reverse=True)
    leaderboard = [r for _, r in scored]
    _write_json(leaderboard_json, leaderboard)

    if leaderboard:
        best = leaderboard[0]
        _write_json(best_json, best)
        print("Best trial:")
        print(f"  trial_id: {best['trial_id']}")
        print(f"  score:    {best['score']}")
        print(f"  dir:      {best['trial_dir']}")
        print(f"  overrides:{best['overrides']}")
        return 0

    print("No successful trials to rank.")
    return 1


def run_one_trial_main(args: argparse.Namespace) -> int:
    """
    Child-process entrypoint: runs a single training job with overrides and writes trial_result.json.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")

    # Ensure parent locomotion/ imports work in the child process too
    if str(LOCOMOTION_DIR) not in sys.path:
        sys.path.insert(0, str(LOCOMOTION_DIR))

    trial_dir = Path(args.trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    overrides = json.loads(args.overrides_json_inline)
    if not isinstance(overrides, dict):
        raise ValueError("overrides_json_inline must decode to a dict")

    from training_config import TrainingConfig
    from training_helpers import setup_logging
    from train import train_bittle

    logger = setup_logging(trial_dir, level=getattr(__import__("logging"), "INFO"))

    cfg = TrainingConfig(test_mode=args.test)

    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Override '{k}' is not a field on TrainingConfig")
        setattr(cfg, k, v)

    try:
        if cfg.batch_size % cfg.num_minibatches != 0:
            logger.warning("batch_size not divisible by num_minibatches. This may error or reduce efficiency.")
        if cfg.batch_size % cfg.unroll_length != 0:
            logger.warning("batch_size not divisible by unroll_length. This may error or reduce efficiency.")
    except Exception:
        pass

    out = train_bittle(
        config=cfg,
        xml_path=args.xml_path,
        output_dir=trial_dir,
        logger=logger,
    )

    result = {
        "success": bool(out.get("success", False)),
        "overrides": overrides,
        "trial_dir": str(trial_dir),
        "summary": out.get("summary", {}) if out.get("success") else {},
        "error": out.get("error"),
    }
    _write_json(trial_dir / "trial_result.json", result)

    return 0 if result["success"] else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_path", type=str, default="bittle_adapted_scene.xml")
    ap.add_argument("--trials_json", type=str, help="Path to JSON list of hyperparameter override dicts.")
    ap.add_argument("--base_output_dir", type=str, default=None)
    ap.add_argument("--metric", type=str, default="best_reward", choices=["best_reward", "final_reward", "mean_reward"])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--_run_one_trial", action="store_true")
    ap.add_argument("--trial_id", type=int, default=0)
    ap.add_argument("--trial_dir", type=str, default="")
    ap.add_argument("--overrides_json_inline", type=str, default="{}")
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()

    if args._run_one_trial:
        return run_one_trial_main(args)

    if not args.trials_json:
        print("ERROR: --trials_json is required for sweep mode.")
        return 2

    return sweep_main(args)


if __name__ == "__main__":
    raise SystemExit(main())