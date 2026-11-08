"""v19: v18 retrained with realistic ball gaps (train_v19.py), applied to our match.

The model (train_v19.py) learned from 7 open Bundesliga matches what possession and dead balls
look like in tracking data. Here it reads our match's tracking, gives each frame a probability
of H / A / D, and the decoder (with the fees chosen on the external matches) turns those into
whole possessions and stoppages.

Nothing is trained, tuned or checked against our match's answer key or video. The report uses
only the provided data.

Usage:
    python solution/train_v19.py   # once, makes solution/models/v19.joblib
    python solution/v19.py         # writes evaluation/results/v19.csv
"""

import json
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
from tracking import load_tracking  # noqa: E402
import features18  # noqa: E402
import model18  # noqa: E402
import train_v19  # noqa: E402
import v1  # noqa: E402
import v17  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v19.csv")


def segments() -> list:
    """Each half from its kick-off to its last graded frame (kick-off first: direction)."""
    out = []
    for half, first, last in half_video_blocks(GAME):
        kickoff = v1.SECOND_HALF_KICKOFF if half == 2 else 11185
        out.append((max(first, kickoff), last + 1))
    return out


def label_match(t: dict) -> tuple:
    model = joblib.load(train_v19.MODEL_PATH)
    with open(train_v19.PARAMS_PATH) as handle:
        params = json.load(handle)
    segs = segments()
    X, names = features18.build(t, segs, params["cap_s"])
    if names != params["features"]:
        raise ValueError("feature list changed since training: run train_v19.py again")
    score = np.zeros((len(X), 3))
    rows = np.concatenate([np.arange(a, b) for a, b in segs])
    score[rows] = model18.log_probs(model, X[rows])
    path = model18.decode(score, segs, params["switch_fee"], params["dead_fee"])
    labels = np.full(len(X), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        labels[first:last + 1] = "D"                    # before kick-off
    labels[path >= 0] = v1.STATES[path[path >= 0]]
    return labels, params


def main() -> None:
    t = load_tracking()
    labels, params = label_match(t)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"\nMODEL: trained on {params['trained_frames']:,} external frames, "
          f"{len(params['features'])} features; fees switch {params['switch_fee']}, "
          f"dead {params['dead_fee']}; external held-out spellF1 {params['heldout_spell_f1']:.3f}")

    old, _ = v17.label_match(t)
    a, b = v17.label_free_checks(t, old), v17.label_free_checks(t, labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    print("\nLABEL-FREE CHECKS (no answer key)               v17       v19")
    print(f"  possessions, full match                   {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages, full match                     {a['stoppages']:>7}   {b['stoppages']:>7}"
          f"   (published: {v17.PUBLISHED_STOPPAGES_PER_MATCH[0]}-{v17.PUBLISHED_STOPPAGES_PER_MATCH[1]})")
    print(f"  dead ball, % of graded frames             {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%"
          f"   (published: {lo}-{hi}%)")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch labelled dead              {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    lo_s, hi_s = v17.PUBLISHED_STOPPAGES_PER_MATCH
    checks = [
        ("stoppages within 30% of published 109-117", 0.7 * lo_s <= b["stoppages"] <= 1.3 * hi_s),
        ("dead share 35-50%", 35 <= b["dead_pct"] <= 50),
        ("ball off pitch labelled dead >= 90%", b["off_dead"] >= 90),
    ]
    print("\nDECISION RULE (fixed before seeing v19's output; no answer key)")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  -> submit {'v19' if all(ok for _, ok in checks) else 'v17'}")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v19.csv")
    print("Do NOT run run_evaluation.py or diagnose.py.")


if __name__ == "__main__":
    main()
