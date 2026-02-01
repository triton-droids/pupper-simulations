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

    # Check if CUDA_VISIBLE_DEVICES is already set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        existing_value = os.environ["CUDA_VISIBLE_DEVICES"]
        visible_devices = [d.strip() for d in existing_value.split(",") if d.strip()]
        num_visible = len(visible_devices)
        print(f"CUDA_VISIBLE_DEVICES already set: {existing_value}", file=sys.stderr)
        print(f"Detected {num_visible} pre-configured devices", file=sys.stderr)
    else:
        # Try to detect GPU count using nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--list-gpus"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_lines = [line for line in result.stdout.strip().split("\n") if line]
                num_visible = len(gpu_lines)
                visible_devices = [str(i) for i in range(num_visible)]
                print(f"nvidia-smi detected {num_visible} GPUs", file=sys.stderr)
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

    if num_to_use > 0 and num_to_use < num_visible:
        # Limit to first num_to_use devices
        devices_to_use = ",".join(visible_devices[:num_to_use])
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_to_use
        print(f"✓ Limiting to {num_to_use}/{num_visible} GPUs (max divisible by 4)", file=sys.stderr)
        print(f"✓ Set CUDA_VISIBLE_DEVICES={devices_to_use}", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return num_to_use
    elif num_to_use == 0:
        # Less than 4 devices, use what we have
        print(f"⚠ Only {num_visible} GPUs available (less than 4) - using all", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return num_visible
    else:
        # num_to_use == num_visible, perfect multiple of 4
        print(f"✓ Using all {num_visible} GPUs (already divisible by 4)", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return num_visible
