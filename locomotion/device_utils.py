"""
Device Management Utilities
============================

Utilities for managing JAX device allocation and limiting.
"""

import os
import subprocess
import sys
from typing import Optional


def setup_device_limit() -> Optional[int]:
    """
    Limit CUDA devices to maximum number divisible by 4.

    This function must be called BEFORE importing JAX to take effect.
    It sets the CUDA_VISIBLE_DEVICES environment variable to limit
    which GPUs are visible to JAX.

    Returns:
        Number of devices that will be used, or None if detection failed
    """
    print("=" * 80, file=sys.stderr)
    print("DEVICE SETUP", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # First, check what CUDA_VISIBLE_DEVICES is currently set to
    initial_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if initial_cuda_env is not None:
        print(f"Initial CUDA_VISIBLE_DEVICES: {initial_cuda_env}", file=sys.stderr)
    else:
        print("CUDA_VISIBLE_DEVICES not set initially", file=sys.stderr)

    # Determine how many devices are currently visible
    # NOTE: nvidia-smi always shows ALL physical GPUs, not what CUDA sees!
    # So we need to respect existing CUDA_VISIBLE_DEVICES if set
    if initial_cuda_env is not None and initial_cuda_env.strip():
        # Parse the existing setting
        visible_devices = [d.strip() for d in initial_cuda_env.split(",") if d.strip()]
        num_visible = len(visible_devices)
        print(f"Using pre-configured devices: {num_visible} devices", file=sys.stderr)
    else:
        # No existing setting, query nvidia-smi for physical GPU count
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_indices = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                num_visible = len(gpu_indices)
                visible_devices = [str(i) for i in range(num_visible)]
                print(f"nvidia-smi detected {num_visible} physical GPUs", file=sys.stderr)
            else:
                print(f"nvidia-smi failed with return code {result.returncode}", file=sys.stderr)
                print(f"stderr: {result.stderr}", file=sys.stderr)
                print("Skipping device limiting - letting JAX auto-detect", file=sys.stderr)
                print("=" * 80, file=sys.stderr)
                return None
        except FileNotFoundError:
            print("nvidia-smi not found - skipping device limiting", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            return None
        except subprocess.TimeoutExpired:
            print("nvidia-smi timeout - skipping device limiting", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            return None

    # Calculate maximum number divisible by 4
    num_to_use = (num_visible // 4) * 4

    if num_to_use == 0:
        # Less than 4 devices, use what we have
        print(f"⚠ Only {num_visible} devices available (less than 4) - using all", file=sys.stderr)
        # Don't override if already set
        if initial_cuda_env is None:
            devices_to_use = ",".join(visible_devices)
            os.environ["CUDA_VISIBLE_DEVICES"] = devices_to_use
            print(f"✓ Set CUDA_VISIBLE_DEVICES={devices_to_use}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return num_visible
    elif num_to_use < num_visible:
        # Need to limit devices
        devices_to_use = ",".join(visible_devices[:num_to_use])
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_to_use
        print(f"✓ Limiting to {num_to_use}/{num_visible} devices (max divisible by 4)", file=sys.stderr)
        print(f"✓ Set CUDA_VISIBLE_DEVICES={devices_to_use}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return num_to_use
    else:
        # num_to_use == num_visible, perfect multiple of 4
        # ALWAYS set it explicitly to ensure we have control
        devices_to_use = ",".join(visible_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_to_use
        print(f"✓ Using all {num_visible} devices (divisible by 4)", file=sys.stderr)
        print(f"✓ Set CUDA_VISIBLE_DEVICES={devices_to_use}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return num_visible
