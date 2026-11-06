"""v6: v5 with ball speed computed from positions, and pass detection.

Changes from v5:
  - Ball speed: the tracking's own ball-speed column is mostly missing and implausibly low
    (90th percentile 3.7 m/s; a pass travels at 10-20 m/s). v6 computes it from the ball's
    positions over +/-2 frames (0.16 s), which smooths tracking jitter.
  - Live-ball clue uses that speed: > 5 m/s on the pitch = being played; 1-5 m/s within
    2 m of a player = dribbled.
  - Pass detection: a "control" is a run of frames where the ball is within 2 m of one
    team's player and slower than 7 m/s. When two consecutive controls belong to the same
    team, less than 3 s apart, the ball travelled between teammates: that team keeps
    possession during the flight, and the ball is live. This follows CHALLENGE.md's own
    proxy for control ("a team completes ... consecutive passes").

Usage:
    python solution/v6.py   # writes evaluation/results/v6.csv
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
from tracking import FPS, HOME, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v3  # noqa: E402
import v4  # noqa: E402
import v5  # noqa: E402
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v6.csv")

SPEED_SPAN = 2              # frames either side for the centred speed estimate
FAST_BALL_MS = 5.0          # faster = being passed or shot
DRIBBLE_MS = 1.0            # 1-5 m/s within control radius = dribbled
CONTROL_MAX_MS = 7.0        # a controlled ball is slower than this
CONTROL_MIN_FRAMES = 3      # shortest run that counts as a control
PASS_MAX_S = 3.0            # longest flight between two controls that counts as a pass
PASS_TEAM_CLUE = 1.0        # per frame of flight, for the passing team
PASS_LIVE_CLUE = 0.5        # per frame of flight, subtracted from D


def ball_speed(t: dict) -> np.ndarray:
    """Ball speed (m/s) from positions: centred difference over 2 * SPEED_SPAN frames."""
    k = SPEED_SPAN
    speed = np.full(len(t["bx"]), np.nan)
    dx = t["bx"][2 * k:] - t["bx"][:-2 * k]
    dy = t["by"][2 * k:] - t["by"][:-2 * k]
    speed[k:-k] = np.hypot(dx, dy) * FPS / (2 * k)
    return speed


def live_ball_clues(t: dict, signals: dict, speed: np.ndarray) -> tuple:
    """Negative D score for a ball being played (fast) or dribbled (moving, at feet)."""
    score = np.zeros((len(speed), 3))
    s = np.nan_to_num(speed, nan=0.0)
    on_pitch = ~np.isnan(t["bx"]) & ~signals["off"]
    fast = on_pitch & (s > FAST_BALL_MS)
    dribble = on_pitch & ~fast & (s > DRIBBLE_MS) & (signals["dist"] <= v1.CONTROL_RADIUS_M)
    score[fast, D] -= v4.FAST_BALL_CLUE
    score[dribble, D] -= v4.AT_FEET_CLUE
    return score, fast, dribble


def controls(team: np.ndarray, dist: np.ndarray, speed: np.ndarray) -> list:
    """[(team, start, stop)] runs where one team's player has the ball close and slow."""
    s = np.nan_to_num(speed, nan=np.inf)
    owner = np.where((dist <= v1.CONTROL_RADIUS_M) & (s < CONTROL_MAX_MS), team, -1)
    change = np.flatnonzero(owner[1:] != owner[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(owner)]))
    return [(int(owner[a]), a, b) for a, b in zip(starts, stops)
            if owner[a] >= 0 and b - a >= CONTROL_MIN_FRAMES]


def pass_clues(t: dict, signals: dict, speed: np.ndarray) -> tuple:
    """Team and live clues over the flight of every same-team pass. Returns score and counts."""
    score = np.zeros((len(speed), 3))
    ctrl = controls(signals["team"], signals["dist"], speed)
    passes = {H: 0, A: 0}
    changes = 0
    for (team_a, _, stop_a), (team_b, start_b, _) in zip(ctrl, ctrl[1:]):
        if start_b - stop_a > PASS_MAX_S * FPS:
            continue
        if team_a == team_b:
            state = H if team_a == HOME else A
            score[stop_a:start_b, state] += PASS_TEAM_CLUE
            score[stop_a:start_b, D] -= PASS_LIVE_CLUE
            passes[state] += 1
        else:
            changes += 1
    return score, passes, changes, len(ctrl)


def label_match(t: dict) -> tuple:
    base, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    n = len(base)
    speed = ball_speed(t)
    tags = load_tags()
    goal_score, goal_kickoffs = v3.goal_clues(n, tags)
    vanish, candidates, confirmed = v3.vanish_clues(t, signals["speed"])
    live, fast, dribble = live_ball_clues(t, signals, speed)
    passing, passes, changes, n_controls = pass_clues(t, signals, speed)
    parts = {
        "set pieces": v4.set_piece_clues(n, tags)[0],
        "goal restarts": goal_score,
        "ball vanished": vanish,
    }
    raw = v1.decode_halves(base + sum(parts.values()) + live + passing)
    labels, fixed = v5.enforce_min_durations(raw)
    stats = dict(candidates=candidates, confirmed=confirmed, goal_kickoffs=goal_kickoffs,
                 fast=fast, dribble=dribble, passes=passes, changes=changes,
                 controls=n_controls, speed=speed, **fixed)
    return labels, parts, signals, stats


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
    print(f"  live ball (dribbled):    {100 * stats['dribble'][graded].mean():.1f}% of graded frames")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")

    print("\nPASSES (whole match; a typical pro match has ~800-1,100)")
    print(f"  controls found:          {stats['controls']}")
    print(f"  same-team passes:        {stats['passes'][H] + stats['passes'][A]} "
          f"(home {stats['passes'][H]}, away {stats['passes'][A]})")
    print(f"  ball changed team:       {stats['changes']}")

    seen = graded & ~np.isnan(stats["speed"])
    raw_present = graded & ~np.isnan(tracking["bx"])
    pct = np.percentile(stats["speed"][seen], [10, 25, 50, 75, 90])
    print("\nBALL SPEED, COMPUTED FROM POSITIONS (graded frames)")
    print("  10% / 25% / 50% / 75% / 90%:  " + " / ".join(f"{p:.2f}" for p in pct) + " m/s")
    print(f"  tracking's own speed column present on {100 * np.mean(~np.isnan(tracking['bs'][raw_present])):.1f}%"
          " of frames where the ball is seen")

    v1.report(labels)
    v3.dead_ball_sources(labels, parts, signals)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v6.csv")


if __name__ == "__main__":
    main()
