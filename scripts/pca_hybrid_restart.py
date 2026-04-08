"""
Wrapper entrypoint for PCA-space hybrid restart selection.

Usage:
  python scripts/pca_hybrid_restart.py --config-module combined_2d --max-episode 10 --top-restarts 10
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

if "--space" not in sys.argv:
    sys.argv[1:1] = ["--space", "pca"]

from analysis.hybrid_restart_selection import main


if __name__ == "__main__":
    main()
