"""v26: v25 rebuilt on external data: a model that also reads Smart Tagging, with the Laws and
Smart Tagging as weighted evidence whose strengths were tuned on the 7 DFL matches.

  - Model (tune_v26.py): v19's features plus Smart Tagging windows, the restarts the Laws
    describe (restarts22), goals, Law 9 timing and 5 s / 10 s context; trained only on DFL
    (stand-in Smart Tagging from DFL events).
  - Decision (pipeline26.py): model + weighted "dead" evidence (the seconds before each
    set-piece restart, goals), decoded with fees; then the v5 + v20 cleanup. Law 9 (ball out)
    always forces a stoppage. The other settings come from v26_params.json, tuned on external
    data only.

No answer key, no video.

Usage:
    python solution/tune_v26.py   # once: model + settings from the external matches
    python solution/v26.py        # writes evaluation/results/v26.csv
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
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import load_tracking  # noqa: E402
import features26  # noqa: E402
import model18  # noqa: E402
import pipeline26 as P  # noqa: E402
import restarts22  # noqa: E402
import tune_v26  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v6  # noqa: E402
import v17  # noqa: E402
import v19  # noqa: E402
import v20  # noqa: E402
import v25  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v26.csv")


def label_match(t: dict) -> tuple:
    model = joblib.load(tune_v26.MODEL_PATH)
    with open(tune_v26.PARAMS_PATH) as handle:
        params = json.load(handle)
    segs = v19.segments()
    tags = load_tags()
    _, signals = v1.clues(t)
    off = signals["off"]
    found = restarts22.find_restarts(t, v6.ball_speed(t), off, tags, v2._norm)
    X, names = features26.build(t, segs, tags, found, off, params["cap_s"])
    if names != params["features"]:
        raise ValueError("feature list changed since training: run tune_v26.py again")
    n = len(X)
    rows = np.concatenate([np.arange(a, b) for a, b in segs])
    log_probs = np.zeros((n, 3))
    log_probs[rows] = model18.log_probs(model, X[rows])
    m = P.masks(t, tags, found, off, v2._norm, v2.SET_PIECE_TYPES)
    codes = P.label(log_probs, m, segs, params["settings"])
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        labels[first:last + 1] = "D"                      # before kick-off
    labels[codes >= 0] = v1.STATES[codes[codes >= 0]]
    return labels, params


def main() -> None:
    t = load_tracking()
    labels, params = label_match(t)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"\nSETTINGS (tuned on external data): {params['settings']}")
    print(f"EXTERNAL spellF1: fair {params['fair_spell_f1']:.3f} (tuned {params['tuned_spell_f1']:.3f}); "
          f"v25: 0.481")

    ref, _ = v25.label_match(t)
    a, b = v17.label_free_checks(t, ref), v17.label_free_checks(t, labels)
    oa, ob = v25.off_dead_1s(t, ref), v25.off_dead_1s(t, labels)
    sa, sb = v20.stoppage_lengths(ref), v20.stoppage_lengths(labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    lo_s, hi_s = v17.PUBLISHED_STOPPAGES_PER_MATCH
    print("\nLABEL-FREE CHECKS (no answer key)               v25       v26")
    print(f"  possessions                               {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages (published {lo_s}-{hi_s})             {a['stoppages']:>7}   {b['stoppages']:>7}")
    print(f"  dead % (published {lo}-{hi})              {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch 1 s+ labelled dead         {oa:>6.1f}%   {ob:>6.1f}%")
    print(f"  median stoppage (pub. mean 23.1 s)        {np.median(sa):>6.1f}s   {np.median(sb):>6.1f}s")
    checks = [
        ("stoppages within 30% of published 109-117", 0.7 * lo_s <= b["stoppages"] <= 1.3 * hi_s),
        ("dead share 35-50%", 35 <= b["dead_pct"] <= 50),
        ("ball off pitch 1 s+ labelled dead >= 90%", ob >= 90),
        ("visible agreement no worse than v23 (83.6%) - 1 point", b["visible_agree"] >= 83.6 - 1),
    ]
    print("\nELIGIBILITY (fixed before v26's output; no answer key)")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    eligible = all(ok for _, ok in checks)
    beats = params["fair_spell_f1"] > 0.481
    print(f"  -> v26 is {'ELIGIBLE' if eligible else 'NOT eligible'}; fair external spellF1 "
          f"{params['fair_spell_f1']:.3f} {'beats' if beats else 'does not beat'} v25's 0.481")
    print(f"  -> final version: {'v26' if eligible and beats else 'v25'}")
    v1.report(labels)
    print("\nNext:  python solution/check_format.py evaluation/results/v26.csv")


if __name__ == "__main__":
    main()
