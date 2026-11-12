"""Robustness of the final pipeline on the external DFL matches.

Each match is held out from training (leave-one-match-out) and degraded at several levels of
ball visibility; the decoder uses the final tuned settings. The provided match is degraded at
roughly 41% missing ball with a typical gap of 10.7 s, so that row matches its conditions.

Note: the decoder settings were tuned on these matches, so absolute scores are slightly
optimistic; the comparison across levels is the point of this test.

Usage:
    python solution/evaluate_external.py      # about 30-40 minutes
"""

import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import external, model, pipeline, training  # noqa: E402

LOG = logging.getLogger("evaluate_external")
SCENARIOS = (
    ("20% ball missing", 0.20, 1.0),
    ("41% ball missing (provided match)", external.BALL_MISSING_SHARE, 1.0),
    ("60% ball missing", 0.60, 1.0),
    ("41% missing, gaps twice as long", external.BALL_MISSING_SHARE, 2.0),
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.time()
    _, settings = pipeline.load_model()
    rng = np.random.default_rng(model.SEED)

    LOG.info("1. Leave-one-match-out classifiers")
    classifiers = {}
    for match_id in external.MATCHES:
        others = [m for m in external.MATCHES if m != match_id]
        X, y, _ = training.training_rows(others, rng)
        classifiers[match_id] = model.new_model().fit(X, y)
        LOG.info("  %s done (%.0f s)", match_id, time.time() - start)

    LOG.info("\n2. Scenarios")
    results = {}
    for name, share, scale in SCENARIOS:
        folds = []
        for match_id in external.MATCHES:
            folds.append(training.held_out(match_id, classifiers[match_id], share, scale))
            LOG.info("  %s | %s (%.0f s)", name, match_id, time.time() - start)
        results[name] = training.evaluate(folds, settings["settings"])

    LOG.info("\n3. Results (mean over 7 held-out matches)")
    LOG.info(training.SUMMARY_HEADER)
    for name, _, _ in SCENARIOS:
        LOG.info(training.summary_row(name, results[name]))
    LOG.info("\n(%.0f s)", time.time() - start)


if __name__ == "__main__":
    main()
