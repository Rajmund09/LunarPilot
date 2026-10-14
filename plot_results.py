"""
LunarPilot - Root Entrypoint for Plotting Figures
Allows running `python plot_results.py` directly from project root.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from lunar_pilot.plot_results import generate_all_plots

if __name__ == "__main__":
    generate_all_plots()
