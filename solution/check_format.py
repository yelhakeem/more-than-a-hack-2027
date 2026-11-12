"""Validate a submission file's format with the organiser's own validation function.

A file that passes here is accepted by the official scorer. This checks the format only
(row count, frame index, labels); it does not read any ground truth.

Usage:
    python solution/check_format.py evaluation/results/submission.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import possession  # noqa: E402,F401  (adds the organiser's modules to the path)
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402


def check(path: str) -> int:
    """Print OK or FAIL with the reason; return a process exit code (0 = valid)."""
    expected = GAMES[GAME]["n_video_frames"]
    try:
        reason = pm.validate_submission(pm.read_label_csv(path), expected)
    except Exception as err:  # unreadable or malformed file
        reason = f"could not read: {err}"
    if reason:
        print(f"FAIL  {os.path.basename(path)}: {reason}")
        return 1
    print(f"OK    {os.path.basename(path)}: {expected:,} rows, frames 0..{expected - 1}, labels H/A/D/X")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(check(sys.argv[1]))
