"""v10: v9 with sharper stoppage edges and the kick-off team.

Changes from v9:
  - Kick-off team: the team named on each "Kick Off (Start)" tag has the ball at the
    restart. v2-v9 ignored these two tags, so nothing said who kicks off each half.
  - Stoppage start from the ball's path: when the ball was last seen heading towards a line
    and then left the pitch or vanished, the moment it crossed is estimated from its speed
    and distance to the line. The stoppage starts there, not seconds later when players
    slow down. Only for exits confirmed by players slowing within 5 s.
  - Back-fill before set pieces: a whistle stops play instantly, but the tag only covers
    the last few seconds before the restart. Walking back from each set-piece tag, the
    stoppage is extended over the quiet stretch before it (no runners and no fast ball),
    up to 40 s.

Usage:
    python solution/v10.py   # writes evaluation/results/v10.csv
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
import v5  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
import v9  # noqa: E402
from v1 import D  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v10.csv")

KICKOFF_TEAM_CLUE = 2.0
KICKOFF_CLUE_S = 3.0
CROSSING_MAX_S = 2.0        # extrapolate the ball's path at most this far
CROSSING_CLUE = 2.0
CROSSING_CLUE_S = 3.0       # dead clue from the crossing for this long
SLOWDOWN_WINDOW_S = 5.0
BACKFILL_MAX_S = 40.0
BACKFILL_CLUE = 1.0


def build_v9_total(t: dict, signals: dict) -> np.ndarray:
    """v9's total clue score, built exactly as v9.label_match does."""
    total = v9.v8_total(t, signals)
    t_filled, filled = v9.fill_short_gaps(t)
    total += v9.fill_clues(t_filled, filled)
    missing = np.isnan(t["bx"]) & ~filled
    total += v9.shape_clues(t, signals, missing)[0]
    return total


def kickoff_clues(n: int, tags: list) -> tuple:
    """Team clue at each match and second-half kick-off, from the tag's Team."""
    score = np.zeros((n, 3))
    used = 0
    for frame, tag in v7.restart_frames(tags):
        if v2._norm(tag["tags"].get("Type")) != "kick off (start)":
            continue
        side = v2.team_side(tag["tags"].get("Team"))
        if side is None:
            continue
        score[frame:min(n, frame + int(KICKOFF_CLUE_S * FPS)), side] += KICKOFF_TEAM_CLUE
        used += 1
    return score, used


def crossing_clues(t: dict, signals: dict, speed: np.ndarray) -> tuple:
    """Dead clue from the estimated moment the ball crossed a line, for confirmed exits."""
    n = len(speed)
    score = np.zeros((n, 3))
    length, width = (float(v) for v in t["pitch"])
    seen = ~np.isnan(t["bx"])
    out_or_missing = signals["off"] | ~seen
    starts = np.flatnonzero(out_or_missing[1:] & ~out_or_missing[:-1]) + 1
    k = v6.SPEED_SPAN
    used = 0
    for e in starts:
        last = e - 1
        if last < k or not seen[last] or not seen[last - k]:
            continue
        # Backward velocity: the centred speed needs frames after the exit, which are missing.
        vx = (t["bx"][last] - t["bx"][last - k]) * FPS / k
        vy = (t["by"][last] - t["by"][last - k]) * FPS / k
        if np.hypot(vx, vy) < 1.0:
            continue
        window = signals["speed"][e:min(n, e + int(SLOWDOWN_WINDOW_S * FPS))]
        if window.size == 0 or window.min() >= v1.SLOW_SPEED_MS:
            continue                                    # play carried on: not a stoppage
        times = []
        if vx > 0:
            times.append((length - t["bx"][last]) / vx)
        if vx < 0:
            times.append(-t["bx"][last] / vx)
        if vy > 0:
            times.append((width - t["by"][last]) / vy)
        if vy < 0:
            times.append(-t["by"][last] / vy)
        times = [x for x in times if 0 <= x <= CROSSING_MAX_S]
        if not times:
            continue
        cross = last + int(round(min(times) * FPS))
        score[cross:min(n, cross + int(CROSSING_CLUE_S * FPS)), D] += CROSSING_CLUE
        used += 1
    return score, used


def backfill_clues(t: dict, signals: dict, speed: np.ndarray, runners: np.ndarray, tags: list) -> tuple:
    """Extend each set-piece stoppage back over the quiet stretch before its tag."""
    n = len(speed)
    score = np.zeros((n, 3))
    fast = np.nan_to_num(speed, nan=0.0) > v6.FAST_BALL_MS
    quiet = ~runners & ~fast
    extended = []
    for frame, tag in v7.restart_frames(tags):
        kind = v2._norm(tag["tags"].get("Type"))
        if kind not in v2.SET_PIECE_TYPES:
            continue
        start = tag["start_frame"]
        lo = max(0, start - int(BACKFILL_MAX_S * FPS))
        i = start
        while i > lo and quiet[i - 1]:
            i -= 1
        if i < start:
            score[i:frame, D] += BACKFILL_CLUE
            extended.append((start - i) / FPS)
    return score, extended


def label_match(t: dict) -> tuple:
    base, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    n = len(base)
    speed = v6.ball_speed(t)
    tags = load_tags()
    runners = v8.runners_present(t)

    total = build_v9_total(t, signals)
    kick, kickoffs = kickoff_clues(n, tags)
    cross, crossings = crossing_clues(t, signals, speed)
    back, extended = backfill_clues(t, signals, speed, runners, tags)
    labels, fixed = v5.enforce_min_durations(v1.decode_halves(total + kick + cross + back))
    stats = dict(kickoffs=kickoffs, crossings=crossings, extended=extended, **fixed)
    return labels, stats


def main() -> None:
    tracking = load_tracking()
    labels, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    ext = np.array(stats["extended"]) if stats["extended"] else np.array([0.0])
    print("\nNEW CLUES USED")
    print(f"  kick-offs with a team:   {stats['kickoffs']}  (should be 2)")
    print(f"  line crossings timed:    {stats['crossings']}")
    print(f"  set pieces back-filled:  {len(stats['extended'])}, median {np.median(ext):.1f} s, "
          f"longest {ext.max():.1f} s")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v10.csv")
    print("then:  python solution/diagnose.py evaluation/results/v10.csv")


if __name__ == "__main__":
    main()
