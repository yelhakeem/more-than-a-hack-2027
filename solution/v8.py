"""v8: v7 plus three rules: last touch for every ball-out, shots keep possession, runners mean live.

Changes from v7:
  - Ball out of play anywhere: the restart goes to the opponent of the last toucher, so a
    deflection out off the team in possession hands the ball to the other team. v7 did this
    for tagged throw-ins only; v8 also does it where the tracking shows the ball leaving the
    pitch and no set-piece tag is nearby (Smart Tagging misses some restarts).
  - Shots: a ball struck at >= 15 m/s within 30 m of the goal the shooter's team attacks,
    heading towards that goal line, stays the shooter's team's possession during its flight,
    and the "everyone walking" clue cannot call it dead while it flies (players stop to
    watch a shot). The flight ends when the ball slows, crosses the goal line or 3 s pass.
  - Runners: play is live whenever at least 2 players are running (> 4 m/s). The team
    average hides one or two players chasing a long ball. Where runners are present the
    walking clue is cancelled, and a fast ball only counts as "kicked away during a
    stoppage" if nobody is running now and 3 s later.

Usage:
    python solution/v8.py   # writes evaluation/results/v8.csv
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
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, HOME, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v4  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v8.csv")

# Runners
RUNNING_MS = 4.0
MIN_RUNNERS = 2
RUNNER_SMOOTH_S = 0.5
# Untagged ball-out restarts
MIN_OUT_S = 1.0             # ball must stay out or missing this long after the exit
TAG_CLEARANCE_S = 20.0      # skip exits this close to a set-piece tag (v4/v7 handle those)
RESTART_SEARCH_S = 60.0     # look this far ahead for the restart touch
RESTART_CLUE_S = 3.0
RESTART_TEAM_CLUE = 1.0
# Shots
SHOT_MIN_MS = 15.0
SHOT_RANGE_M = 30.0
SHOT_MAX_FLIGHT_S = 3.0
FLIGHT_END_MS = 5.0
SHOT_TEAM_CLUE = 1.0
SHOT_LIVE_CLUE = 2.0        # subtracted from D during the flight


def runners_present(t: dict) -> np.ndarray:
    """True where at least MIN_RUNNERS players run faster than RUNNING_MS (0.5 s smoothing)."""
    count = np.nansum(t["ps"] > RUNNING_MS, axis=1).astype(float)
    k = int(RUNNER_SMOOTH_S * FPS)
    count = np.convolve(count, np.ones(k) / k, mode="same")
    return count >= MIN_RUNNERS


def walking_clue(speed: np.ndarray) -> np.ndarray:
    """v1's walking contribution to D, per frame (to cancel it where runners are present)."""
    out = np.zeros(len(speed))
    slow = speed < v1.SLOW_SPEED_MS
    out[slow] = v1.SLOW_CLUE * np.minimum((v1.SLOW_SPEED_MS - speed[slow]) / v1.SLOW_SPEED_MS * 2, 2)
    return out


def live_ball_clues(t: dict, signals: dict, speed: np.ndarray, restarts: list,
                    runners: np.ndarray) -> tuple:
    """v6's live-ball clue; a fast ball is 'kicked away' only if nobody runs now or 3 s later."""
    n = len(speed)
    score, fast, dribble = v6.live_ball_clues(t, signals, speed)
    shift = int(v7.STILL_WALKING_AFTER_S * FPS)
    runners_later = np.concatenate((runners[shift:], np.repeat(runners[-1:], shift)))
    after_restart = np.zeros(n, dtype=bool)
    for frame, _ in restarts:
        after_restart[frame:min(n, frame + int(v7.AFTER_RESTART_S * FPS))] = True
    kicked_away = fast & ~runners & ~runners_later & ~after_restart
    score[kicked_away, D] += v4.FAST_BALL_CLUE
    return score, fast & ~kicked_away, dribble, kicked_away


def last_toucher(signals: dict, exit_frame: int):
    """Team of the last toucher before exit_frame (v7's rule), or None if unseen or contested."""
    a = max(0, exit_frame - int(v7.LAST_TOUCH_S * FPS))
    touches = np.flatnonzero(signals["dist"][a:exit_frame] <= v7.TOUCH_RADIUS_M) + a
    if touches.size == 0:
        return None
    recent = touches[touches >= touches[-1] - int(v7.CONTEST_S * FPS)]
    teams = set(signals["team"][recent].tolist())
    return teams.pop() if len(teams) == 1 else None


def ball_out_clues(t: dict, signals: dict, tags: list) -> tuple:
    """Restart-team clue for ball-outs the tagging missed: the opponent of the last toucher."""
    n = len(t["bx"])
    score = np.zeros((n, 3))
    length, width = (float(v) for v in t["pitch"])
    m = v2.LINE_MARGIN_M
    seen = ~np.isnan(t["bx"])
    near_line = seen & ((t["bx"] < m) | (t["bx"] > length - m) | (t["by"] < m) | (t["by"] > width - m))
    out_or_missing = signals["off"] | ~seen
    exit_start = np.zeros(n, dtype=bool)
    exit_start[1:] = out_or_missing[1:] & ~out_or_missing[:-1] & (signals["off"][1:] | near_line[:-1])

    tag_frames = np.array([tg["start_frame"] for tg in tags if v2._norm(tg["code"]) == "set pieces"])
    min_out = int(MIN_OUT_S * FPS)
    exits = resolved = 0
    for e in np.flatnonzero(exit_start):
        if e + min_out >= n or not out_or_missing[e:e + min_out].all():
            continue
        if tag_frames.size and np.min(np.abs(tag_frames - e)) <= TAG_CLEARANCE_S * FPS:
            continue
        exits += 1
        toucher = last_toucher(signals, e)
        if toucher is None:
            continue
        stop = min(n, e + int(RESTART_SEARCH_S * FPS))
        back_in = np.flatnonzero(~out_or_missing[e:stop] & (signals["dist"][e:stop] <= v1.CONTROL_RADIUS_M))
        if back_in.size == 0:
            continue
        restart = e + back_in[0]
        state = A if toucher == HOME else H
        score[restart:min(n, restart + int(RESTART_CLUE_S * FPS)), state] += RESTART_TEAM_CLUE
        resolved += 1
    return score, exits, resolved


def shot_clues(t: dict, signals: dict, speed: np.ndarray) -> tuple:
    """Shooter's-team and live clues over the flight of every detected shot."""
    n = len(speed)
    score = np.zeros((n, 3))
    length = float(t["pitch"][0])
    ends = v7.defended_ends(t)
    s = np.nan_to_num(speed, nan=0.0)
    k = v6.SPEED_SPAN
    vx = np.zeros(n)
    vx[k:-k] = np.nan_to_num(t["bx"][2 * k:] - t["bx"][:-2 * k], nan=0.0) * FPS / (2 * k)

    shots = 0
    for half, first, last in half_video_blocks(GAME):
        i = first + 1
        while i <= last:
            if not (s[i] >= SHOT_MIN_MS and s[i - 1] < SHOT_MIN_MS):
                i += 1
                continue
            toucher = last_toucher(signals, i)
            if toucher is None:
                i += 1
                continue
            home_defends_x0 = ends[half]
            attacks_xl = home_defends_x0 if toucher == HOME else not home_defends_x0
            goal_x = length if attacks_xl else 0.0
            towards = vx[i] > 0 if attacks_xl else vx[i] < 0
            if not towards or abs(goal_x - t["bx"][i]) > SHOT_RANGE_M:
                i += 1
                continue
            stop = min(last + 1, i + int(SHOT_MAX_FLIGHT_S * FPS))
            for j in range(i, stop):
                over_line = t["bx"][j] > length if attacks_xl else t["bx"][j] < 0
                if (s[j] < FLIGHT_END_MS and j > i + 2) or over_line:
                    stop = j
                    break
            state = H if toucher == HOME else A
            score[i:stop, state] += SHOT_TEAM_CLUE
            score[i:stop, D] -= SHOT_LIVE_CLUE
            shots += 1
            i = max(stop, i + 1)
    return score, shots


def label_match(t: dict) -> tuple:
    base, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    n = len(base)
    speed = v6.ball_speed(t)
    tags = load_tags()
    restarts = v7.restart_frames(tags)
    runners = runners_present(t)

    base[runners, D] -= walking_clue(signals["speed"])[runners]     # runners cancel "walking"
    goal_score, goal_kickoffs = v3.goal_clues(n, tags)
    vanish, candidates, confirmed = v3.vanish_clues(t, signals["speed"])
    live, fast, dribble, kicked_away = live_ball_clues(t, signals, speed, restarts, runners)
    passing, passes, changes, n_controls = v6.pass_clues(t, signals, speed)
    throw_score, throw_ins, throw_resolved = v7.throw_in_clues(t, signals, restarts)
    keeper, catches, ends = v7.keeper_clues(t, signals, speed)
    out_score, exits, exits_resolved = ball_out_clues(t, signals, tags)
    shot_score, shots = shot_clues(t, signals, speed)
    parts = {
        "set pieces": v4.set_piece_clues(n, tags)[0],
        "goal restarts": goal_score,
        "ball vanished": vanish,
    }
    total = (base + sum(parts.values()) + live + passing + throw_score + keeper
             + out_score + shot_score)
    labels, fixed = v5.enforce_min_durations(v1.decode_halves(total))
    stats = dict(fast=fast, kicked_away=kicked_away, dribble=dribble, runners=runners,
                 passes=passes, throw_ins=throw_ins, throw_resolved=throw_resolved,
                 catches=catches, exits=exits, exits_resolved=exits_resolved, shots=shots,
                 ends=ends, **fixed)
    return labels, parts, signals, stats


def main() -> None:
    tracking = load_tracking()
    labels, parts, signals, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))

    print("\nCLUES USED")
    print(f"  2+ players running:      {100 * stats['runners'][graded].mean():.1f}% of graded frames")
    print(f"  live ball (fast):        {100 * stats['fast'][graded].mean():.1f}% of graded frames")
    print(f"  fast but kicked away:    {100 * stats['kicked_away'][graded].mean():.1f}% of graded frames")
    print(f"  same-team passes:        {stats['passes'][H] + stats['passes'][A]}")
    print(f"  shots detected:          {stats['shots']}  (a typical match has roughly 20-30)")
    print(f"  throw-ins:               {stats['throw_ins']}, {stats['throw_resolved']} with a last toucher found")
    print(f"  untagged ball-outs:      {stats['exits']}, {stats['exits_resolved']} with a last toucher found")
    print(f"  keeper catches:          {stats['catches']}")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")

    v1.report(labels)
    v3.dead_ball_sources(labels, parts, signals)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v8.csv")


if __name__ == "__main__":
    main()
