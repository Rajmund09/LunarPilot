"""
LunarPilot - Root Entrypoint for Training
Allows running `python train.py` directly from project root.
"""

import os
import sys

# Add src to python path for direct invocation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from lunar_pilot.train import train

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LunarPilot PPO Agent")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config file")
    args = parser.parse_args()
    train(config_path=args.config)
