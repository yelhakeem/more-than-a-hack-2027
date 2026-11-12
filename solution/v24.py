"""v24: the external-data model (v19) decides every frame; the Laws and Smart Tagging overrule it.

On the 7 held-out Bundesliga matches (external_eval.py) the model alone scored far higher than
every rules version (spellF1 0.548 vs v23's 0.354), but on our match it called a ball clearly
off the pitch dead only 43% of the time. So:

  1. Start from v19's labels.
  2. Law 9: a ball more than 1 m off the pitch for at least 1 s is out of play. Dead from the
     moment it leaves until it is next seen on the pitch.
  3. Smart Tagging set pieces (allowed input): dead from 4 s before the tag (v2) to the restart
     found by restarts22.py (the Laws' restart: throw back in, still ball struck).
  4. Smart Tagging goals: dead for the 41 s before each "Kick Off (After Goal)" (v17, published
     goal-delay statistics).
  5. The usual cleanup: v5 minimum durations, v20 switch blips.

No answer key, no video; no new numbers.

Usage:
    python solution/v24.py   # writes evaluation/results/v24.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, load_tracking  # noqa: E402
import restarts22  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v8  # noqa: E402
import v17  # noqa: E402
import v19  # noqa: E402
import v20  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v24.csv")
OUT_MIN_S = v8.MIN_OUT_S             # 1 s
GOAL_WINDOW_S = 41.0                 # v17


def ball_out(t: dict, off: np.ndarray) -> np.ndarray:
    """Law 9: off the pitch for 1 s+, dead until the ball is next seen on the pitch."""
    n = len(off)
    seen_on = ~np.isnan(t["bx"]) & ~off
    next_on = np.full(n + 1, n)
    for i in range(n - 1, -1, -1):
        next_on[i] = i if seen_on[i] else next_on[i + 1]
    dead = np.zeros(n, dtype=bool)
    padded = np.concatenate(([0], off.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= OUT_MIN_S * FPS:
            dead[a:next_on[b]] = True
    return dead


def tag_dead(n: int, tags: list, found: dict) -> np.ndarray:
    """Set pieces: dead from 4 s before the tag to its restart. Goals: 41 s before the kick-off."""
    dead = np.zeros(n, dtype=bool)
    for tag in tags:
        kind = v2._norm(tag["tags"].get("Type"))
        if v2._norm(tag["code"]) == "set pieces" and kind in v2.SET_PIECE_TYPES and id(tag) in found:
            restart = found[id(tag)][0]
            dead[max(0, min(tag["start_frame"], restart) - int(v2.BEFORE_TAG_S * FPS)):restart] = True
        if kind == "kick off (after goal)":
            dead[max(0, tag["start_frame"] - int(GOAL_WINDOW_S * FPS)):tag["start_frame"]] = True
    return dead


def label_match(t: dict) -> tuple:
    model, _ = v19.label_match(t)
    _, signals = v1.clues(t)
    tags = load_tags()
    found = restarts22.find_restarts(t, v6.ball_speed(t), signals["off"], tags, v2._norm)
    out = ball_out(t, signals["off"])
    sets = tag_dead(len(model), tags, found)
    labels = model.copy()
    graded = labels != "X"
    labels[(out | sets) & graded] = "D"
    labels, _ = v5.enforce_min_durations(labels)
    labels, _, _ = v20.remove_switch_blips(labels, signals["off"])
    changed = int(np.sum(labels != model))
    return labels, dict(out=int((out & graded).sum()), sets=int((sets & graded).sum()), changed=changed)


def main() -> None:
    t = load_tracking()
    labels, stats = label_match(t)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"\nOVERRULED: ball out (Law 9) {stats['out']:,} frames, set pieces and goals "
          f"{stats['sets']:,} frames; {stats['changed']:,} frames differ from v19")

    import v23  # noqa: E402  (only for the comparison column)
    import gap_study  # noqa: E402
    import json
    with open(gap_study.PARAMS_PATH) as handle:
        ref = v23.label_match(t, json.load(handle)["min_gap_s"])[1]
    a, b = v17.label_free_checks(t, ref), v17.label_free_checks(t, labels)
    sa, sb = v20.stoppage_lengths(ref), v20.stoppage_lengths(labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    lo_s, hi_s = v17.PUBLISHED_STOPPAGES_PER_MATCH
    print("\nLABEL-FREE CHECKS (no answer key)               v23       v24")
    print(f"  possessions                               {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages (published {lo_s}-{hi_s})             {a['stoppages']:>7}   {b['stoppages']:>7}")
    print(f"  dead % (published {lo}-{hi})              {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch labelled dead              {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    print(f"  median stoppage (pub. mean 23.1 s)        {np.median(sa):>6.1f}s   {np.median(sb):>6.1f}s")
    checks = [
        ("stoppages within 30% of published 109-117", 0.7 * lo_s <= b["stoppages"] <= 1.3 * hi_s),
        ("dead share 35-50%", 35 <= b["dead_pct"] <= 50),
        ("ball off pitch labelled dead >= 90%", b["off_dead"] >= 90),
        ("visible agreement no worse than v23 - 1 point", b["visible_agree"] >= a["visible_agree"] - 1),
    ]
    print("\nELIGIBILITY (fixed before v24's output; no answer key)")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  -> v24 is {'ELIGIBLE' if all(ok for _, ok in checks) else 'NOT eligible'}; "
          f"then the higher external spellF1 (external_eval.py) wins: v23 0.354")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v24.csv")
    print("then:  python solution/external_eval.py   (now includes v24)")


if __name__ == "__main__":
    main()
