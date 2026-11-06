"""v12: v11 with back-fill only from a real trigger, and slowly carried balls.

Changes from v11:
  - Back-fill needs a trigger. A stoppage starts with an event: the ball leaving the pitch,
    or the whistle, when all players slow down together. v12 back-fills a set piece only
    from the latest such trigger in the 30 s before its tag, so slow build-up play with no
    trigger is never swallowed (v10/v11 back-filled any quiet stretch).
  - Slowly carried ball: v11's "still ball" (under 0.5 m/s) is widened to a ball moving
    under 3 m/s, i.e. walked or jogged to the restart spot, while nobody runs and the team
    average is walking pace. The near-player and dribble clues are removed there and a
    small dead-ball clue is added.

Usage:
    python solution/v12.py   # writes evaluation/results/v12.csv
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
import v4  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402
import v11  # noqa: E402
from v1 import D  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v12.csv")

BACKFILL_MAX_S = 30.0
SLOWDOWN_DROP_MS = 1.0      # team average falls by at least this much...
SLOWDOWN_SPAN_S = 1.0       # ...between 1 s before and 1 s after the frame...
SLOWDOWN_TO_MS = 1.5        # ...to below this: everyone stopped together (the whistle)
CARRIED_MAX_MS = 3.0
CARRIED_DEAD_CLUE = 1.0


def triggers(t: dict, signals: dict) -> np.ndarray:
    """Frames where a stoppage can start: the ball leaves the pitch, or everyone slows at once."""
    n = len(t["bx"])
    length, width = (float(v) for v in t["pitch"])
    m = v2.LINE_MARGIN_M
    seen = ~np.isnan(t["bx"])
    near_line = seen & ((t["bx"] < m) | (t["bx"] > length - m) | (t["by"] < m) | (t["by"] > width - m))
    out_or_missing = signals["off"] | ~seen
    exit_start = np.zeros(n, dtype=bool)
    exit_start[1:] = out_or_missing[1:] & ~out_or_missing[:-1] & (signals["off"][1:] | near_line[:-1])

    k = int(SLOWDOWN_SPAN_S * FPS)
    speed = signals["speed"]
    slowdown = np.zeros(n, dtype=bool)
    slowdown[k:-k] = ((speed[:-2 * k] - speed[2 * k:] >= SLOWDOWN_DROP_MS)
                      & (speed[2 * k:] < SLOWDOWN_TO_MS))
    # keep only the first frame of each slowdown run
    slowdown[1:] &= ~slowdown[:-1]
    return exit_start | slowdown


def backfill_clues(t: dict, signals: dict, tags: list) -> tuple:
    """Dead clue from the latest trigger before each set-piece tag up to its restart."""
    n = len(t["bx"])
    score = np.zeros((n, 3))
    trig = np.flatnonzero(triggers(t, signals))
    restarts = sorted(v7.restart_frames(tags), key=lambda r: r[0])
    extended, previous_restart = [], 0
    for frame, tag in restarts:
        start = tag["start_frame"]
        if v2._norm(tag["tags"].get("Type")) in v2.SET_PIECE_TYPES:
            lo = max(previous_restart, start - int(BACKFILL_MAX_S * FPS))
            found = trig[(trig >= lo) & (trig < start)]
            if found.size:
                score[found[-1]:frame, D] += v10.BACKFILL_CLUE
                extended.append((start - found[-1]) / FPS)
        previous_restart = frame
    return score, extended


def adjustment_clues(t: dict, signals: dict, speed: np.ndarray, runners: np.ndarray) -> tuple:
    """v11's duel rule, plus the widened still/carried-ball rule."""
    near = v11.near_player_clue(signals)
    home_d, away_d = v11.nearest_by_team(t)
    contested = (home_d <= v1.CONTROL_RADIUS_M) & (away_d <= v1.CONTROL_RADIUS_M)
    s = np.nan_to_num(speed, nan=np.inf)
    on_pitch = ~np.isnan(t["bx"]) & ~signals["off"]
    carried = (on_pitch & (s < CARRIED_MAX_MS) & ~runners
               & (signals["speed"] < v1.SLOW_SPEED_MS) & (signals["dist"] <= v1.NEAR_RADIUS_M))
    dribble = (on_pitch & (s > v6.DRIBBLE_MS) & (s <= v6.FAST_BALL_MS)
               & (signals["dist"] <= v1.CONTROL_RADIUS_M))
    score = np.zeros_like(near)
    score[contested | carried] -= near[contested | carried]
    score[carried, D] += CARRIED_DEAD_CLUE
    score[carried & dribble, D] += v4.AT_FEET_CLUE        # undo v6's dribble push against D
    return score, contested, carried


def label_match(t: dict) -> tuple:
    _, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    speed = v6.ball_speed(t)
    tags = load_tags()
    runners = v8.runners_present(t)

    total = v10.build_v9_total(t, signals)
    cross, crossings = v10.crossing_clues(t, signals, speed)
    back, extended = backfill_clues(t, signals, tags)
    adjust, contested, carried = adjustment_clues(t, signals, speed, runners)
    labels, fixed = v5.enforce_min_durations(v1.decode_halves(total + cross + back + adjust))
    stats = dict(crossings=crossings, extended=extended, contested=contested, carried=carried, **fixed)
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
    print(f"  ball carried, all walk:  {100 * stats['carried'][graded].mean():.1f}% of graded frames")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v12.csv")
    print("then:  python solution/diagnose.py evaluation/results/v12.csv")


if __name__ == "__main__":
    main()
