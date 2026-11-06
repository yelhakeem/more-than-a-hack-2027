"""v11: v10 with safer back-filling, contested balls and still balls.

Changes from v10:
  - Kick-off rule removed: both "Kick Off (Start)" tags carry Team = N/A, so it never fired.
  - Back-fill before set pieces now stops at any sign of live play: the ball seen moving on
    the pitch (> 2 m/s) or the previous restart. In v10 a slow build-up with nobody running
    counted as "quiet" and was back-filled as dead for up to 40 s. Cap lowered to 30 s.
  - Contested ball: when players from BOTH teams are within 2 m of the ball it is a duel, so
    the "ball near a player" clue goes to neither team and the last team keeps possession
    (CHALLENGE.md: contested or loose balls stay with the team that had them).
  - Still ball: a ball that is barely moving, next to a player, while nobody runs and the
    team average is walking pace, is being placed or collected for a restart. The "ball near
    a player" clue is removed there and a small dead-ball clue added.

Usage:
    python solution/v11.py   # writes evaluation/results/v11.csv
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, HOME, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v11.csv")

BACKFILL_MAX_S = 30.0
LIVE_BALL_MS = 2.0          # ball moving at least this fast on the pitch = play is live
STILL_BALL_MS = 0.5
STILL_DEAD_CLUE = 1.0


def nearest_by_team(t: dict) -> tuple:
    """Distance (m) from the ball to the nearest home player and to the nearest away player."""
    dist = np.hypot(t["px"] - t["bx"][:, None], t["py"] - t["by"][:, None])
    home = t["team"] == HOME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmin(dist[:, home], axis=1), np.nanmin(dist[:, ~home], axis=1)


def near_player_clue(signals: dict) -> np.ndarray:
    """v1's ball-near-player contribution, per frame and state (to remove it where needed)."""
    score = np.zeros((len(signals["dist"]), 3))
    dist, team = signals["dist"], signals["team"]
    strong = dist <= v1.CONTROL_RADIUS_M
    weak = (dist > v1.CONTROL_RADIUS_M) & (dist <= v1.NEAR_RADIUS_M)
    for state, side in ((H, HOME), (A, 1 - HOME)):
        score[strong & (team == side), state] += v1.STRONG_CLUE
        score[weak & (team == side), state] += v1.WEAK_CLUE
    return score


def adjustment_clues(t: dict, signals: dict, speed: np.ndarray, runners: np.ndarray) -> tuple:
    """Remove the near-player clue in duels and around still balls; add dead clue for still balls."""
    near = near_player_clue(signals)
    home_d, away_d = nearest_by_team(t)
    contested = (home_d <= v1.CONTROL_RADIUS_M) & (away_d <= v1.CONTROL_RADIUS_M)
    still = (~np.isnan(t["bx"]) & ~signals["off"]
             & (np.nan_to_num(speed, nan=np.inf) < STILL_BALL_MS)
             & ~runners & (signals["speed"] < v1.SLOW_SPEED_MS)
             & (signals["dist"] <= v1.NEAR_RADIUS_M))
    score = np.zeros_like(near)
    score[contested | still] -= near[contested | still]
    score[still, D] += STILL_DEAD_CLUE
    return score, contested, still


def backfill_clues(t: dict, signals: dict, speed: np.ndarray, runners: np.ndarray, tags: list) -> tuple:
    """v10's back-fill, stopped by a moving ball on the pitch or the previous restart."""
    n = len(speed)
    score = np.zeros((n, 3))
    s = np.nan_to_num(speed, nan=0.0)
    live = (~np.isnan(t["bx"]) & ~signals["off"] & (s > LIVE_BALL_MS)) | runners
    restarts = sorted(v7.restart_frames(tags), key=lambda r: r[0])
    extended = []
    previous_restart = 0
    for frame, tag in restarts:
        kind = v2._norm(tag["tags"].get("Type"))
        start = tag["start_frame"]
        if kind in v2.SET_PIECE_TYPES:
            lo = max(previous_restart, start - int(BACKFILL_MAX_S * FPS))
            i = start
            while i > lo and not live[i - 1]:
                i -= 1
            if i < start:
                score[i:frame, D] += v10.BACKFILL_CLUE
                extended.append((start - i) / FPS)
        previous_restart = frame
    return score, extended


def label_match(t: dict) -> tuple:
    base, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    speed = v6.ball_speed(t)
    tags = load_tags()
    runners = v8.runners_present(t)

    total = v10.build_v9_total(t, signals)
    cross, crossings = v10.crossing_clues(t, signals, speed)
    back, extended = backfill_clues(t, signals, speed, runners, tags)
    adjust, contested, still = adjustment_clues(t, signals, speed, runners)
    labels, fixed = v5.enforce_min_durations(v1.decode_halves(total + cross + back + adjust))
    stats = dict(crossings=crossings, extended=extended, contested=contested, still=still, **fixed)
    return labels, stats


def main() -> None:
    tracking = load_tracking()
    labels, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))
    ext = np.array(stats["extended"]) if stats["extended"] else np.array([0.0])
    print("\nNEW CLUES USED")
    print(f"  line crossings timed:    {stats['crossings']}")
    print(f"  set pieces back-filled:  {len(stats['extended'])}, median {np.median(ext):.1f} s, "
          f"longest {ext.max():.1f} s")
    print(f"  contested ball (duels):  {100 * stats['contested'][graded].mean():.1f}% of graded frames")
    print(f"  still ball, all walking: {100 * stats['still'][graded].mean():.1f}% of graded frames")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v11.csv")
    print("then:  python solution/diagnose.py evaluation/results/v11.csv")


if __name__ == "__main__":
    main()
