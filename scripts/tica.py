"""
Wrapper entrypoint for phosphate-pathway TICA on PPO trajectories.

Usage:
  python scripts/tica.py --config-module combined_2d --max-traj 0 --lag 5
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.tica_phosphate_pathway import main


if __name__ == "__main__":
    main()
