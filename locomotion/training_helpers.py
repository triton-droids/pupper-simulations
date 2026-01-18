import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

from flax.training import orbax_utils
from orbax import checkpoint as ocp

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """
    Set up comprehensive logging to both console and file.
    
    Args:
        output_dir: Directory for log files
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger
    """
    # Create logs directory
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('bittle_training')
    logger.setLevel(level)
    
    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with detailed formatting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(log_dir / f'training_{timestamp}.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def policy_params_callback(output_dir: Path, logger: logging.Logger, monitor=None):
    """
    Create a callback for saving policy checkpoints.

    Args:
        output_dir: Base output directory
        logger: Logger instance
        monitor: TrainingMonitor instance to update with inference function

    Returns:
        Callback function
    """
    ckpt_path = (output_dir / 'checkpoints').resolve()
    ckpt_path.mkdir(parents=True, exist_ok=True)

    def callback(current_step: int, make_policy: Any, params: Any) -> None:
        """Save checkpoint at current step."""
        # Update monitor with inference function for video generation
        if monitor is not None and monitor.make_inference_fn_cached is None:
            monitor.make_inference_fn_cached = make_policy(params)

        orbax_checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        path = ckpt_path / f'step_{current_step:08d}'
        orbax_checkpointer.save(path, params, force=True, save_args=save_args)
        logger.info(f"Saved checkpoint to {path}")

    return callback

# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Bittle quadruped locomotion policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python train_bittle.py --test
  
  # Full training run
  python train_bittle.py
  
  # Custom output directory
  python train_bittle.py --output_dir ./experiments/run_001
        """
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (minimal training for fast iteration)'
    )
    
    parser.add_argument(
        '--xml_path',
        type=str,
        default='bittle_adapted_scene.xml',
        help='Path to MuJoCo XML scene file (default: bittle_adapted_scene.xml)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for checkpoints and logs (default: auto-generated)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()