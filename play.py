"""
LunarPilot - Root Entrypoint for Playback / Demo
Allows running `python play.py` directly from project root.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from lunar_pilot.play import play

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Demo for LunarPilot PPO Agent")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--fps", type=int, default=50, help="Frames per second")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions stochastically")
    parser.add_argument("--random", action="store_true", help="Visualize random policy")
    args = parser.parse_args()

    play(
        model_path=args.model,
        num_episodes=args.episodes,
        fps=args.fps,
        deterministic=not args.stochastic,
        random_policy=args.random,
    )
