"""
Device Management Utilities
============================

Utilities for managing JAX device allocation and limiting.
"""

import os
import subprocess
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
    # Check if CUDA_VISIBLE_DEVICES is already set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        num_visible = len(visible_devices)
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
            else:
                # Can't detect GPUs, let JAX handle it
                return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # nvidia-smi not available or timeout, let JAX handle it
            return None

    # Calculate maximum number divisible by 4
    num_to_use = (num_visible // 4) * 4

    if num_to_use > 0 and num_to_use < num_visible:
        # Limit to first num_to_use devices
        devices_to_use = ",".join(visible_devices[:num_to_use])
        os.environ["CUDA_VISIBLE_DEVICES"] = devices_to_use
        print(
            f"Device limiting: {num_visible} GPUs available, using {num_to_use} (max divisible by 4)"
        )
        return num_to_use
    elif num_to_use == 0:
        # Less than 4 devices, use what we have
        print(
            f"Device limiting: Only {num_visible} GPUs available (less than 4), using all"
        )
        return num_visible
    else:
        # num_to_use == num_visible, perfect multiple of 4
        print(f"Device limiting: Using all {num_visible} GPUs (divisible by 4)")
        return num_visible
