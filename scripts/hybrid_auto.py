"""
Wrapper entrypoint for closed-loop hybrid PPO training.

Usage:
  python scripts/hybrid_auto.py --rounds 2 --initial-episodes 10 --episodes-per-round 10
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.hybrid_auto_controller import main


if __name__ == "__main__":
    main()
