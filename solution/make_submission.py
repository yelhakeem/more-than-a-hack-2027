"""Produce the submission file for the provided match.

Loads the match tracking and Smart Tagging, applies the trained classifier and decoder, writes
the 'frame,label' CSV and validates its format with the organiser's own check.

Usage:
    python solution/make_submission.py                       # evaluation/results/submission.csv
    python solution/make_submission.py --out path/to/team.csv
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import config, data, pipeline  # noqa: E402
from check_format import check  # noqa: E402

LOG = logging.getLogger("make_submission")


def _runs(flags: np.ndarray) -> int:
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flags.astype(np.int8), [0]))))
    return len(edges) // 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=config.DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    clf, settings = pipeline.load_model()
    labels = pipeline.label_match(data.load_tracking(), data.load_tags(), clf, settings)
    data.write_submission(args.out, labels)
    LOG.info("Wrote %s", os.path.relpath(args.out))

    graded = labels != "X"
    possessions = sum(_runs(labels[first:last + 1] == s)
                      for _, first, last in data.graded_blocks() for s in "HA")
    stoppages = sum(_runs(labels[first:last + 1] == "D") for _, first, last in data.graded_blocks())
    LOG.info("  graded frames: %d | H %.1f%% | A %.1f%% | D %.1f%%", graded.sum(),
             *(100 * np.mean(labels[graded] == s) for s in "HAD"))
    LOG.info("  possessions: %d | stoppages: %d", possessions, stoppages)
    return check(args.out)


if __name__ == "__main__":
    sys.exit(main())
