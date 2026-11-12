"""v25: v24 with its hard evidence protected from the cleanup.

v24 forced dead balls from Law 9 (ball over 1 m off the pitch for 1 s or more) and from Smart
Tagging (set pieces, goals), then ran v5's minimum-duration cleanup, which could turn a 1-3 s
Law 9 stoppage back into play. v25 re-applies the forced dead balls after the cleanup, so the
Laws always have the last word. Nothing else changes; no new numbers.

Off-pitch check, amended on Nov 8 before v25 was run: it counts only balls off the pitch for at
least 1 s (the Law 9 threshold v8 uses), not sub-second blips that are mostly tracking noise
or a ball in the air (about two-thirds come straight back onto the pitch within 1 s).

No answer key, no video.

Usage:
    python solution/v25.py   # writes evaluation/results/v25.csv
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, graded_mask, load_tracking  # noqa: E402
import gap_study  # noqa: E402
import restarts22  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v17  # noqa: E402
import v19  # noqa: E402
import v20  # noqa: E402
import v24  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v25.csv")


def label_match(t: dict) -> tuple:
    model, _ = v19.label_match(t)
    _, signals = v1.clues(t)
    tags = load_tags()
    found = restarts22.find_restarts(t, v6.ball_speed(t), signals["off"], tags, v2._norm)
    forced = v24.ball_out(t, signals["off"]) | v24.tag_dead(len(model), tags, found)
    graded = model != "X"
    labels = model.copy()
    labels[forced & graded] = "D"
    labels, _ = v5.enforce_min_durations(labels)
    labels, _, _ = v20.remove_switch_blips(labels, signals["off"])
    restored = int(np.sum((labels != "D") & forced & graded))
    labels[forced & graded] = "D"                       # the Laws have the last word
    return labels, dict(forced=int((forced & graded).sum()), restored=restored)


def off_dead_1s(t: dict, labels: np.ndarray) -> float:
    """Share of frames with the ball over 1 m off the pitch, in a run of 1 s+, labelled dead."""
    _, signals = v1.clues(t)
    off = signals["off"] & graded_mask(len(labels))
    keep = np.zeros(len(off), dtype=bool)
    padded = np.concatenate(([0], off.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= v24.OUT_MIN_S * FPS:
            keep[a:b] = True
    return 100 * float(np.mean(labels[keep] == "D")) if keep.any() else float("nan")


def main() -> None:
    t = load_tracking()
    labels, stats = label_match(t)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"\nFORCED DEAD (Law 9 + Smart Tagging): {stats['forced']:,} frames; "
          f"{stats['restored']:,} of them had been undone by the cleanup and were restored")

    import v23  # noqa: E402  (comparison column only)
    with open(gap_study.PARAMS_PATH) as handle:
        ref = v23.label_match(t, json.load(handle)["min_gap_s"])[1]
    a, b = v17.label_free_checks(t, ref), v17.label_free_checks(t, labels)
    oa, ob = off_dead_1s(t, ref), off_dead_1s(t, labels)
    sa, sb = v20.stoppage_lengths(ref), v20.stoppage_lengths(labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    lo_s, hi_s = v17.PUBLISHED_STOPPAGES_PER_MATCH
    print("\nLABEL-FREE CHECKS (no answer key)               v23       v25")
    print(f"  possessions                               {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages (published {lo_s}-{hi_s})             {a['stoppages']:>7}   {b['stoppages']:>7}")
    print(f"  dead % (published {lo}-{hi})              {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch 1 s+ labelled dead         {oa:>6.1f}%   {ob:>6.1f}%")
    print(f"  (old check: any off-pitch frame)          {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    print(f"  median stoppage (pub. mean 23.1 s)        {np.median(sa):>6.1f}s   {np.median(sb):>6.1f}s")
    checks = [
        ("stoppages within 30% of published 109-117", 0.7 * lo_s <= b["stoppages"] <= 1.3 * hi_s),
        ("dead share 35-50%", 35 <= b["dead_pct"] <= 50),
        ("ball off pitch 1 s+ labelled dead >= 90% (amended Nov 8)", ob >= 90),
        ("visible agreement no worse than v23 - 1 point", b["visible_agree"] >= a["visible_agree"] - 1),
    ]
    print("\nELIGIBILITY (fixed before v25's output; no answer key)")
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"  -> v25 is {'ELIGIBLE' if all(ok for _, ok in checks) else 'NOT eligible'}; "
          f"then the higher external spellF1 (external_eval.py) wins")
    v1.report(labels)
    print("\nNext:  python solution/check_format.py evaluation/results/v25.csv")
    print("then:  python solution/external_eval.py   (now includes v25)")


if __name__ == "__main__":
    main()
