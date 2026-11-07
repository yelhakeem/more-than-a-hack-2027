"""v17: v12 with its stoppage timings taken from published match statistics.

Same rules as v12. The three timings that were changed during development while public-window
scores were being looked at are reset to values derived from external data:

  Augste & Cordes (2016), "Game stoppages as a tactical means in soccer", IJPAS 16(3): 1053-64,
  FIFA World Cup 2014, mean +/- SD stoppage duration:
    throw-in 11.5 +/- 7.3 s, free kick 25.2 +/- 19.4 s, goal delay 60.3 +/- 18.9 s,
    all stoppages 23.1 +/- 21.8 s, play stopped 43.3% of the match.

  - VANISH_MAX_S (v3, 15 s): the ball vanishing near a line is mostly a throw-in.
    Cap = throw-in mean + 1 SD = 18.8 -> 19 s.
  - BACKFILL_MAX_S (v12, 30 s): how far before a set-piece tag a stoppage can start.
    Cap = all-stoppage mean + 1 SD = 44.9 -> 45 s. v12 already needs a real trigger
    (ball out or whistle slowdown), so the cap is only an upper limit.
  - BEFORE_GOAL_KICKOFF_S (v3, 45 s): dead time before a kick-off after a goal.
    Goal delay mean - 1 SD = 41.4 -> 41 s, so the window is almost always truly dead.

Rule used for all three: a cap that only limits how far a stoppage can extend uses mean + 1 SD;
a window that forces a stoppage uses mean - 1 SD.

The public 5-minute ground truth is NOT used to judge this version. The report below uses
only the provided data and the published statistics above.

Usage:
    python solution/v17.py   # writes evaluation/results/v17.csv
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, half_video_blocks  # noqa: E402
from tracking import HOME, graded_mask, load_tracking, run_lengths  # noqa: E402
import v1  # noqa: E402
import v3  # noqa: E402
import v11  # noqa: E402
import v12  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v17.csv")

# (module, name, v17 value)
TIMINGS = (
    (v3, "VANISH_MAX_S", 19.0),
    (v12, "BACKFILL_MAX_S", 45.0),
    (v3, "BEFORE_GOAL_KICKOFF_S", 41.0),
)
PUBLISHED_STOPPED_PCT = (41.8, 43.3)    # Augste & Cordes 2016, World Cups 2006 and 2014
PUBLISHED_STOPPAGES_PER_MATCH = (109, 117)
VISIBLE_RADIUS_M = 1.0


def label_match(t: dict, timings=TIMINGS) -> tuple:
    """v12 with the given timings; the modules' original values are restored afterwards."""
    saved = [(module, name, getattr(module, name)) for module, name, _ in timings]
    try:
        for module, name, value in timings:
            setattr(module, name, value)
        return v12.label_match(t)
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def label_free_checks(t: dict, labels: np.ndarray) -> dict:
    """Checks that use only the provided data, never the answer key."""
    n = len(labels)
    graded = graded_mask(n)
    _, signals = v1.clues(t)
    _, team = v1.nearest_player(t)
    home_d, away_d = v11.nearest_by_team(t)
    on_pitch = ~np.isnan(t["bx"]) & ~signals["off"]
    visible = graded & on_pitch & (signals["dist"] <= VISIBLE_RADIUS_M) & (
        np.minimum(home_d, away_d) <= VISIBLE_RADIUS_M) & (
        np.maximum(home_d, away_d) > v1.CONTROL_RADIUS_M)
    visible_label = np.where(team == HOME, "H", "A")
    off = graded & signals["off"]
    poss = stoppages = 0
    for _, first, last in half_video_blocks(GAME):
        part = labels[first:last + 1]
        poss += sum(len(run_lengths(part == s)) for s in "HA")
        stoppages += len(run_lengths(part == "D"))
    return {
        "possessions": poss,
        "stoppages": stoppages,
        "dead_pct": 100 * float(np.mean(labels[graded] == "D")),
        "visible_agree": 100 * float(np.mean(labels[visible] == visible_label[visible])),
        "visible_frames": int(visible.sum()),
        "off_dead": 100 * float(np.mean(labels[off] == "D")) if off.any() else float("nan"),
    }


def main() -> None:
    t = load_tracking()
    labels, stats = label_match(t)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")

    old, _ = v12.label_match(t)
    a, b = label_free_checks(t, old), label_free_checks(t, labels)
    lo, hi = PUBLISHED_STOPPED_PCT
    print("\nTIMINGS (from Augste & Cordes 2016)")
    for module, name, value in TIMINGS:
        print(f"  {name:<24} v12 {getattr(module, name):>5.0f} s  ->  v17 {value:>5.0f} s")
    print("\nLABEL-FREE CHECKS (no answer key)               v12       v17")
    print(f"  possessions, full match                   {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages, full match                     {a['stoppages']:>7}   {b['stoppages']:>7}"
          f"   (published: {PUBLISHED_STOPPAGES_PER_MATCH[0]}-{PUBLISHED_STOPPAGES_PER_MATCH[1]} per match)")
    print(f"  dead ball, % of graded frames             {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%"
          f"   (published: {lo}-{hi}%)")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%"
          f"   ({b['visible_frames']:,} frames)")
    print(f"  ball off pitch labelled dead              {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    print(f"\n  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v17.csv")
    print("Do NOT run run_evaluation.py or diagnose.py on this version.")


if __name__ == "__main__":
    main()
