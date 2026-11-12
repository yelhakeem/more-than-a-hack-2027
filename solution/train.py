"""Train the classifier and tune the decoder on the external DFL matches.

Steps:
    1. Leave-one-match-out: for each of the 7 matches, train on the other 6 and predict its
       held-out copy (degraded to single-camera quality).
    2. Tune the decoder settings on all held-out predictions (reported for reference).
    3. Fair score: re-tune with each match excluded and score that match only.
    4. Train the final classifier on all 7 matches; save it with the tuned settings.

Writes solution/models/model.joblib and solution/models/model.json. Uses external data only.

Usage:
    python solution/train.py              # about 30-45 minutes
"""

import json
import logging
import os
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import config, external, model, training  # noqa: E402

LOG = logging.getLogger("train")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.time()
    rng = np.random.default_rng(model.SEED)

    LOG.info("1. Leave-one-match-out predictions")
    folds = []
    for match_id in external.MATCHES:
        others = [m for m in external.MATCHES if m != match_id]
        X, y, _ = training.training_rows(others, rng)
        clf = model.new_model().fit(X, y)
        folds.append(training.held_out(match_id, clf))
        LOG.info("  %s done (%.0f s)", match_id, time.time() - start)

    LOG.info("\n2. Decoder tuning (all 7 matches)")
    best = training.tune(folds, training.DEFAULT_SETTINGS, log=LOG.info)
    tuned = training.evaluate(folds, best)
    LOG.info("  settings: %s", best)

    LOG.info("\n3. Fair external score (each match scored with settings tuned without it)")
    fair = []
    for k, fold in enumerate(folds):
        settings = training.tune(folds[:k] + folds[k + 1:], training.DEFAULT_SETTINGS)
        fair.append(training.evaluate([fold], settings)[0])
        LOG.info("  %s: spellF1 %.3f (%.0f s)", fold["id"], fair[-1]["spell"], time.time() - start)
    LOG.info(training.SUMMARY_HEADER)
    LOG.info(training.summary_row("fair (reported)", fair))
    LOG.info(training.summary_row("tuned on all 7 (optimistic)", tuned))

    LOG.info("\n4. Final classifier (all 7 matches)")
    X, y, names = training.training_rows(external.MATCHES, rng)
    clf = model.new_model().fit(X, y)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump(clf, config.MODEL_PATH)
    with open(config.SETTINGS_PATH, "w") as handle:
        json.dump({"settings": {k: (float(v) if isinstance(v, float) else v) for k, v in best.items()},
                   "features": names, "cap_s": config.TIME_CAP_S,
                   "fair_spell_f1": training.mean(fair), "tuned_spell_f1": training.mean(tuned)},
                  handle, indent=2)
    LOG.info("  %d frames, %d features; saved %s and %s (%.0f s)", len(y), len(names),
             os.path.relpath(config.MODEL_PATH), os.path.relpath(config.SETTINGS_PATH), time.time() - start)


if __name__ == "__main__":
    main()
