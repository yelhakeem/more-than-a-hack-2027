"""Frame-level ball-possession labelling for the More Than A Hack 2027 challenge.

The pipeline labels every video frame of the provided match as home possession (H), away
possession (A), dead ball (D) or not graded (X):

    tracking + Smart Tagging -> features -> classifier -> decoder -> labels

The classifier and every tuned setting were fitted on external data only (the DFL open
dataset); see README.md.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The organiser's helpers (games.py, starter/, evaluation/) are plain modules, not a package.
for _path in (REPO_ROOT, os.path.join(REPO_ROOT, "starter"), os.path.join(REPO_ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
