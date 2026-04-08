"""
Wrapper entrypoint for automatic CV2 candidate ranking.

Usage:
  python scripts/cv2_autoselect.py --config-module combined_2d --max-traj 0
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.cv2_autoselect import main


if __name__ == "__main__":
    main()
