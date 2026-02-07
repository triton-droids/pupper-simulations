"""Diagnostic logging utilities for comparing training and visualization behavior.

Provides shared logging functionality to capture execution traces for debugging
discrepancies between training video generation and local visualization.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import jax.numpy as jp


class DiagnosticLogger:
    """Logger for capturing execution traces during policy rollouts."""

    def __init__(self, output_path: str):
        """
        Initialize diagnostic logger.

        Args:
            output_path: Path to write JSON log file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.data = {
            "metadata": {},
            "reset": {},
            "rollout": [],
            "summary": {}
        }

    def log_metadata(self, seed: int, num_steps: int, **kwargs):
        """
        Log metadata about the execution.

        Args:
            seed: Random seed used
            num_steps: Number of steps to execute
            **kwargs: Additional metadata fields
        """
        self.data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "seed": int(seed),
            "num_steps": int(num_steps),
            **kwargs
        }

    def log_reset(self, command: Any, initial_obs: Any):
        """
        Log data from environment reset.

        Args:
            command: Command velocity [lin_x, lin_y, ang_z]
            initial_obs: Initial observation (first 34 elements - one timestep)
        """
        # Convert JAX/numpy arrays to lists for JSON serialization
        command_list = self._to_list(command)

        # Extract first 34 elements (one timestep) from observation
        obs_list = self._to_list(initial_obs)
        if len(obs_list) > 34:
            obs_list = obs_list[:34]

        self.data["reset"] = {
            "command": command_list,
            "initial_obs": obs_list
        }

    def log_step(
        self,
        step: int,
        obs: Any,
        action: Any,
        base_pos: Any,
        base_vel: Any,
        joint_pos: Any
    ):
        """
        Log data from a single step.

        Args:
            step: Step number
            obs: Current observation (first 34 elements - one timestep)
            action: Policy action output (9 joint commands)
            base_pos: Base position [x, y, z]
            base_vel: Base velocity [vx, vy, vz]
            joint_pos: Joint positions (9 values)
        """
        # Extract first 34 elements from observation
        obs_list = self._to_list(obs)
        if len(obs_list) > 34:
            obs_list = obs_list[:34]

        step_data = {
            "step": int(step),
            "obs": obs_list,
            "action": self._to_list(action),
            "base_pos": self._to_list(base_pos),
            "base_vel": self._to_list(base_vel),
            "joint_pos": self._to_list(joint_pos)
        }

        self.data["rollout"].append(step_data)

    def log_summary(self, **kwargs):
        """
        Log summary statistics.

        Args:
            **kwargs: Summary statistics (e.g., base_z_min, base_z_max, etc.)
        """
        # Convert all values to basic Python types
        summary = {}
        for key, value in kwargs.items():
            if isinstance(value, (np.ndarray, jp.ndarray)):
                summary[key] = float(value)
            elif isinstance(value, (list, tuple)):
                summary[key] = [float(v) if isinstance(v, (np.ndarray, jp.ndarray)) else v
                               for v in value]
            else:
                summary[key] = value

        self.data["summary"] = summary

    def compute_and_log_summary(self):
        """Compute and log summary statistics from rollout data."""
        if not self.data["rollout"]:
            return

        # Extract trajectories
        base_z_values = [step["base_pos"][2] for step in self.data["rollout"]]
        actions = [step["action"] for step in self.data["rollout"]]

        # Flatten actions for statistics
        all_actions = []
        for action_list in actions:
            all_actions.extend(action_list)

        self.log_summary(
            base_z_min=float(np.min(base_z_values)),
            base_z_max=float(np.max(base_z_values)),
            base_z_mean=float(np.mean(base_z_values)),
            base_z_std=float(np.std(base_z_values)),
            action_min=float(np.min(all_actions)),
            action_max=float(np.max(all_actions)),
            action_mean=float(np.mean(all_actions)),
            action_std=float(np.std(all_actions)),
            total_logged_steps=len(self.data["rollout"])
        )

    def save(self):
        """Save diagnostic log to JSON file."""
        with open(self.output_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def _to_list(self, value: Any) -> List[float]:
        """Convert JAX/numpy array or scalar to list of floats."""
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        elif isinstance(value, (np.ndarray, jp.ndarray)):
            return [float(v) for v in np.array(value).flatten()]
        else:
            return [float(value)]


def create_logger(output_path: str) -> DiagnosticLogger:
    """
    Create a diagnostic logger instance.

    Args:
        output_path: Path to write JSON log file

    Returns:
        DiagnosticLogger instance
    """
    return DiagnosticLogger(output_path)
