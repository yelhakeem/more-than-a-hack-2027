"""v20: v17 without the one-frame stoppages at changes of possession.

The decoder charges 30 to switch team directly but only 12 + 12 = 24 to go through a dead
ball, so at many changes of possession it inserted a stoppage lasting a frame or two (compare.py:
167 of v17's 289 stoppages were under 3 s, median 0.0 s). A real stoppage takes seconds to
restart; the shortest common one, a throw-in, is 11.5 +/- 7.3 s (Augste & Cordes 2016).

Change: a stoppage shorter than 3 s (v5's minimum) between two DIFFERENT teams is a loose ball
between them, and CHALLENGE.md says a loose ball stays with the team that had it. v5 already did
this when the same team was on both sides. Exception: if the ball was clearly off the pitch
(over 1 m) for at least 1 s inside it, it really was out of play (Law 9) and stays dead.

No new numbers; no answer key.

Usage:
    python solution/v20.py   # writes evaluation/results/v20.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
from tracking import FPS, load_tracking, run_lengths  # noqa: E402
import v1  # noqa: E402
import v5  # noqa: E402
import v8  # noqa: E402
import v17  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v20.csv")
OUT_EVIDENCE_S = v8.MIN_OUT_S          # 1 s, as for v8's ball-out rule


def remove_switch_blips(labels: np.ndarray, off: np.ndarray) -> tuple:
    """Short stoppages between two different teams go to the team that had the ball."""
    out = labels.copy()
    fixed = kept = 0
    for _, first, last in half_video_blocks(GAME):
        part = out[first:last + 1]
        part_off = off[first:last + 1]
        segments = v5.runs(part)
        for i in range(1, len(segments) - 1):
            label, a, b = segments[i]
            before, after = segments[i - 1][0], segments[i + 1][0]
            if label != "D" or before not in "HA" or after not in "HA" or before == after:
                continue
            if b - a >= v5.MIN_DEAD_S * FPS:
                continue
            if run_lengths(part_off[a:b]).max(initial=0) >= OUT_EVIDENCE_S * FPS:
                kept += 1
                continue
            part[a:b] = before
            fixed += 1
    return out, fixed, kept


def stoppage_lengths(labels: np.ndarray) -> np.ndarray:
    lengths = []
    for _, first, last in half_video_blocks(GAME):
        lengths += list(run_lengths(labels[first:last + 1] == "D") / FPS)
    return np.array(lengths)


def main() -> None:
    t = load_tracking()
    base, _ = v17.label_match(t)
    _, signals = v1.clues(t)
    labels, fixed, kept = remove_switch_blips(base, signals["off"])
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"\nSWITCH BLIPS: {fixed} removed, {kept} kept (ball clearly out for 1 s+)")

    a, b = v17.label_free_checks(t, base), v17.label_free_checks(t, labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    print("\nLABEL-FREE CHECKS (no answer key)               v17       v20")
    print(f"  possessions, full match                   {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages, full match                     {a['stoppages']:>7}   {b['stoppages']:>7}"
          f"   (published: {v17.PUBLISHED_STOPPAGES_PER_MATCH[0]}-{v17.PUBLISHED_STOPPAGES_PER_MATCH[1]})")
    print(f"  dead ball, % of graded frames             {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%"
          f"   (published: {lo}-{hi}%)")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch labelled dead              {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    sa, sb = stoppage_lengths(base), stoppage_lengths(labels)
    print(f"  stoppages under 3 s                       {int((sa < 3).sum()):>7}   {int((sb < 3).sum()):>7}")
    print(f"  median stoppage                           {np.median(sa):>6.1f}s   {np.median(sb):>6.1f}s"
          f"   (published mean 23.1 s)")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v20.csv")
    print("then:  python solution/compare.py v17 v20")


if __name__ == "__main__":
    main()
