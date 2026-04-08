"""
Wrapper entrypoint for hybrid PPO/TICA restart selection.

Usage:
  python scripts/hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.hybrid_restart_selection import main


if __name__ == "__main__":
    main()
