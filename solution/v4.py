"""v4: v3 with a "live ball" counter-clue and measured set-piece timing.

Changes from v3:
  - Live-ball clue: a ball seen on the pitch and moving fast (a pass or shot), or at a
    player's feet, counts against D. Slow build-up play (players walking while one team
    circulates the ball) was being mistaken for a stoppage by the walking clue.
  - Set-piece timing: the restart sits ~70% of the way through each tag window, measured
    from the two published kick-off frames (11,185 and 113,742) against their kick-off
    tags. v2/v3 assumed the midpoint.
  - Report: ball speed percentiles, to check the fast-ball threshold against the data.

Usage:
    python solution/v4.py   # writes evaluation/results/v4.csv
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
from tracking import FPS, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
from v1 import D  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v4.csv")

# Set-piece restart position inside the tag window, measured on both kick-offs:
# (11185 - 11097) / 125 = 0.70 and (113742 - 113660) / 125 = 0.66
RESTART_FRACTION = 0.68
# Live ball
FAST_BALL_MS = 3.0          # a ball moving faster than a jog is being played
FAST_BALL_CLUE = 1.5        # subtracted from D
AT_FEET_CLUE = 0.5          # subtracted from D when a player is within the control radius


def set_piece_clues(n_frames: int, tags: list) -> tuple:
    """v2's set-piece clue with the restart placed at the measured fraction of the window."""
    score = np.zeros((n_frames, 3))
    counts = {"set pieces": 0, "with team": 0}
    for tag in tags:
        if (v2._norm(tag["code"]) != "set pieces"
                or v2._norm(tag["tags"].get("Type")) not in v2.SET_PIECE_TYPES):
            continue
        start, end = tag["start_frame"], tag["end_frame"]
        restart = start + int(round(RESTART_FRACTION * (end - start)))
        score[max(0, start - int(v2.BEFORE_TAG_S * FPS)):restart, D] += v2.SET_PIECE_DEAD_CLUE
        counts["set pieces"] += 1

        side = v2.team_side(tag["tags"].get("Team"))
        if side is not None:
            score[restart:min(n_frames, end + int(v2.AFTER_TAG_S * FPS)), side] += v2.RESTART_TEAM_CLUE
            counts["with team"] += 1
    return score, counts


def live_ball_clues(t: dict, signals: dict) -> tuple:
    """Negative D score where the ball is on the pitch and moving fast or at a player's feet.

    Returns the score and the two boolean masks, for the report.
    """
    score = np.zeros((len(t["bx"]), 3))
    on_pitch = ~np.isnan(t["bx"]) & ~signals["off"]
    fast = on_pitch & (np.nan_to_num(t["bs"], nan=0.0) > FAST_BALL_MS)
    at_feet = on_pitch & ~fast & (signals["dist"] <= v1.CONTROL_RADIUS_M)
    score[fast, D] -= FAST_BALL_CLUE
    score[at_feet, D] -= AT_FEET_CLUE
    return score, fast, at_feet


def label_match(t: dict) -> tuple:
    base, signals = v1.clues(t)
    n = len(base)
    tags = load_tags()
    goal_score, goal_kickoffs = v3.goal_clues(n, tags)
    vanish, candidates, confirmed = v3.vanish_clues(t, signals["speed"])
    live, fast, at_feet = live_ball_clues(t, signals)
    parts = {
        "set pieces": set_piece_clues(n, tags)[0],
        "goal restarts": goal_score,
        "ball vanished": vanish,
    }
    labels = v1.decode_halves(base + sum(parts.values()) + live)
    stats = dict(candidates=candidates, confirmed=confirmed, goal_kickoffs=goal_kickoffs,
                 fast=fast, at_feet=at_feet)
    return labels, parts, signals, stats


def ball_speed_report(t: dict) -> None:
    """Ball speed percentiles on graded frames where the ball is seen (label-free)."""
    seen = graded_mask(len(t["bs"])) & ~np.isnan(t["bs"])
    pct = np.percentile(t["bs"][seen], [10, 25, 50, 75, 90])
    print("\nBALL SPEED (graded frames where the ball is seen)")
    print("  10% / 25% / 50% / 75% / 90%:  " + " / ".join(f"{p:.2f}" for p in pct) + " m/s")
    print(f"  share above the {FAST_BALL_MS} m/s 'being played' threshold:  "
          f"{100 * np.mean(t['bs'][seen] > FAST_BALL_MS):.1f}%")


def main() -> None:
    tracking = load_tracking()
    labels, parts, signals, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))
    print("\nCLUES USED")
    print(f"  ball vanished near line: {stats['candidates']} stretches, "
          f"{stats['confirmed']} confirmed by a slowdown")
    print(f"  goal kick-offs used:     {stats['goal_kickoffs']}")
    print(f"  live ball (fast):        {100 * stats['fast'][graded].mean():.1f}% of graded frames")
    print(f"  live ball (at feet):     {100 * stats['at_feet'][graded].mean():.1f}% of graded frames")
    v1.report(labels)
    v3.dead_ball_sources(labels, parts, signals)
    ball_speed_report(tracking)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v4.csv")
    print("then:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
