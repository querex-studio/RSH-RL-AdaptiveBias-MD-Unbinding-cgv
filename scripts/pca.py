"""
Wrapper entrypoint for phosphate-pathway PCA on PPO trajectories.

Usage:
  python scripts/pca.py --max-traj 50 --cutoff 6.0 --max-residues 60
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.pca_phosphate_pathway import main


if __name__ == "__main__":
    main()
