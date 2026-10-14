"""
LunarPilot - Root Entrypoint for Evaluation
Allows running `python evaluate.py` directly from project root.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from lunar_pilot.evaluate import evaluate_policy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LunarPilot Policy")
    parser.add_argument("--model", type=str, default="models/best/best_model.zip", help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=5000, help="Random seed for evaluation")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")
    parser.add_argument("--output", type=str, default="results/metrics/ppo_evaluation_metrics.json", help="Output path")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        n_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        output_path=args.output,
    )
