"""v7: v6 plus three football rules: kicked-away balls, throw-in last touch, keeper holding.

Changes from v6:
  - Fast ball: still counts as live play, EXCEPT when players are walking now and are still
    walking 3 s later (the ball being kicked or thrown away during a stoppage). Right after a
    tagged restart a fast ball always counts as live, so goal kicks taken while players are
    still walking back are not mistaken for stoppages.
  - Throw-ins: Smart Tagging gives no team. The throw goes to the opponent of the LAST
    TOUCHER (a deflection or clearance by a defender gives the throw to the attackers). We
    find where the ball left the pitch shortly before the tag, take the nearest player in
    the last 3 s before that, and give the restart to the other team. No guess when the
    ball was not seen near a player, or when both teams were within reach at the end.
  - Keeper holding: a ball arriving fast (shot or cross), then held still inside a penalty
    area next to the defending team's deepest player, is a keeper's catch: live, and the
    keeper's team has it. Which end each team defends is read from the kick-off positions.

Usage:
    python solution/v7.py   # writes evaluation/results/v7.csv
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
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v7.csv")
KICKOFFS = {1: 11185, 2: v1.SECOND_HALF_KICKOFF}   # published in DATA.md and FAQ.md

# Fast ball during a stoppage
STILL_WALKING_AFTER_S = 3.0     # players must still be walking this long after the fast ball
AFTER_RESTART_S = 5.0           # a fast ball this soon after a tagged restart is always live
# Throw-in last touch
EXIT_SEARCH_S = 20.0            # look for the ball leaving the pitch this long before the tag
LAST_TOUCH_S = 3.0              # last touch must be this close before the exit
TOUCH_RADIUS_M = 1.5
CONTEST_S = 0.5                 # both teams within reach in this final stretch = ambiguous
THROW_IN_TEAM_CLUE = 1.0
# Keeper holding
PENALTY_DEPTH_M = 16.5
PENALTY_HALF_WIDTH_M = 20.16
HELD_MAX_MS = 1.0               # ball this slow = held or at rest
ARRIVAL_MIN_MS = 8.0            # ball arrived at least this fast (a shot or cross)
ARRIVAL_WINDOW_S = 2.0
KEEPER_HOLD_S = 6.0             # keeper typically holds the ball this long
KEEPER_TEAM_CLUE = 1.0
KEEPER_LIVE_CLUE = 1.5          # subtracted from D


def restart_frames(tags: list) -> list:
    """[(restart_frame, tag)] for every set piece, restart placed as in v4."""
    out = []
    for tag in tags:
        if v2._norm(tag["code"]) != "set pieces":
            continue
        start, end = tag["start_frame"], tag["end_frame"]
        out.append((start + int(round(v4.RESTART_FRACTION * (end - start))), tag))
    return out


def live_ball_clues(t: dict, signals: dict, speed: np.ndarray, restarts: list) -> tuple:
    """v6's live-ball clue, minus fast balls kicked away during a stoppage."""
    n = len(speed)
    score, fast, dribble = v6.live_ball_clues(t, signals, speed)

    walking = signals["speed"] < v1.SLOW_SPEED_MS
    shift = int(STILL_WALKING_AFTER_S * FPS)
    walking_later = np.concatenate((walking[shift:], np.repeat(walking[-1:], shift)))
    after_restart = np.zeros(n, dtype=bool)
    for frame, _ in restarts:
        after_restart[frame:min(n, frame + int(AFTER_RESTART_S * FPS))] = True

    kicked_away = fast & walking & walking_later & ~after_restart
    score[kicked_away, D] += v4.FAST_BALL_CLUE      # undo v6's push against D on these frames
    return score, fast & ~kicked_away, dribble, kicked_away


def throw_in_clues(t: dict, signals: dict, restarts: list) -> tuple:
    """Restarting-team clue for throw-ins: the opponent of the last toucher before the exit."""
    n = len(t["bx"])
    score = np.zeros((n, 3))
    length, width = (float(v) for v in t["pitch"])
    m = v2.LINE_MARGIN_M
    seen = ~np.isnan(t["bx"])
    near_line = seen & ((t["bx"] < m) | (t["bx"] > length - m) | (t["by"] < m) | (t["by"] > width - m))
    # The ball "exits" on its first off-pitch frame, or where it vanishes right after a line.
    vanished = np.zeros(n, dtype=bool)
    vanished[1:] = ~seen[1:] & near_line[:-1]
    exits = signals["off"] | vanished

    throw_ins = resolved = 0
    for frame, tag in restarts:
        if v2._norm(tag["tags"].get("Type")) != "throw-in":
            continue
        throw_ins += 1
        lo = max(0, tag["start_frame"] - int(EXIT_SEARCH_S * FPS))
        found = np.flatnonzero(exits[lo:tag["start_frame"]])
        if found.size == 0:
            continue
        exit_frame = lo + found[0]
        a = max(0, exit_frame - int(LAST_TOUCH_S * FPS))
        touches = np.flatnonzero(signals["dist"][a:exit_frame] <= TOUCH_RADIUS_M) + a
        if touches.size == 0:
            continue
        last = touches[-1]
        recent = touches[touches >= last - int(CONTEST_S * FPS)]
        teams = set(signals["team"][recent].tolist())
        if len(teams) != 1:
            continue                                    # both teams within reach: ambiguous
        toucher = teams.pop()
        restart_state = A if toucher == HOME else H     # the other team throws
        end = min(n, tag["end_frame"] + int(v2.AFTER_TAG_S * FPS))
        score[frame:end, restart_state] += THROW_IN_TEAM_CLUE
        resolved += 1
    return score, throw_ins, resolved


def defended_ends(t: dict) -> dict:
    """{half: True if HOME defends the x = 0 end}, from mean positions just after kick-off."""
    ends = {}
    home = t["team"] == HOME
    for half, kickoff in KICKOFFS.items():
        xs = t["px"][kickoff:kickoff + FPS]
        ends[half] = bool(np.nanmean(xs[:, home]) < np.nanmean(xs[:, ~home]))
    return ends


def keeper_clues(t: dict, signals: dict, speed: np.ndarray) -> tuple:
    """Live + keeper's-team clue for a fast-arriving ball then held in a box by the deepest defender."""
    n = len(speed)
    score = np.zeros((n, 3))
    length, width = (float(v) for v in t["pitch"])
    ends = defended_ends(t)
    s = np.nan_to_num(speed, nan=np.inf)
    in_width = np.abs(t["by"] - width / 2) <= PENALTY_HALF_WIDTH_M
    near_x0 = in_width & (t["bx"] <= PENALTY_DEPTH_M)
    near_xl = in_width & (t["bx"] >= length - PENALTY_DEPTH_M)

    window = int(ARRIVAL_WINDOW_S * FPS)
    padded = np.concatenate((np.zeros(window), np.nan_to_num(speed, nan=0.0)))
    recent_max = np.lib.stride_tricks.sliding_window_view(padded, window)[:-1].max(axis=1)

    catches = 0
    hold_until = -1
    for half, first, last in half_video_blocks(GAME):
        home_at_x0 = ends[half]
        for i in range(first, last + 1):
            if i < hold_until or s[i] > HELD_MAX_MS or signals["dist"][i] > TOUCH_RADIUS_M:
                continue
            if recent_max[i] < ARRIVAL_MIN_MS:
                continue
            team = signals["team"][i]
            defends_x0 = home_at_x0 if team == HOME else not home_at_x0
            if not (near_x0[i] if defends_x0 else near_xl[i]):
                continue
            mates = t["px"][i, t["team"] == team]
            holder_x = t["px"][i][np.nanargmin(np.hypot(t["px"][i] - t["bx"][i], t["py"][i] - t["by"][i]))]
            deepest = np.nanmin(mates) if defends_x0 else np.nanmax(mates)
            if holder_x != deepest:
                continue
            state = H if team == HOME else A
            stop = min(last + 1, i + int(KEEPER_HOLD_S * FPS))
            score[i:stop, state] += KEEPER_TEAM_CLUE
            score[i:stop, D] -= KEEPER_LIVE_CLUE
            hold_until = stop
            catches += 1
    return score, catches, ends


def label_match(t: dict) -> tuple:
    base, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    n = len(base)
    speed = v6.ball_speed(t)
    tags = load_tags()
    restarts = restart_frames(tags)

    goal_score, goal_kickoffs = v3.goal_clues(n, tags)
    vanish, candidates, confirmed = v3.vanish_clues(t, signals["speed"])
    live, fast, dribble, kicked_away = live_ball_clues(t, signals, speed, restarts)
    passing, passes, changes, n_controls = v6.pass_clues(t, signals, speed)
    throw_score, throw_ins, resolved = throw_in_clues(t, signals, restarts)
    keeper, catches, ends = keeper_clues(t, signals, speed)
    parts = {
        "set pieces": v4.set_piece_clues(n, tags)[0],
        "goal restarts": goal_score,
        "ball vanished": vanish,
    }
    raw = v1.decode_halves(base + sum(parts.values()) + live + passing + throw_score + keeper)
    labels, fixed = v5.enforce_min_durations(raw)
    stats = dict(candidates=candidates, confirmed=confirmed, goal_kickoffs=goal_kickoffs,
                 fast=fast, dribble=dribble, kicked_away=kicked_away, passes=passes,
                 changes=changes, throw_ins=throw_ins, resolved=resolved, catches=catches,
                 ends=ends, **fixed)
    return labels, parts, signals, stats


def main() -> None:
    tracking = load_tracking()
    labels, parts, signals, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))

    print("\nCLUES USED")
    print(f"  live ball (fast):        {100 * stats['fast'][graded].mean():.1f}% of graded frames")
    print(f"  fast but kicked away:    {100 * stats['kicked_away'][graded].mean():.1f}% of graded frames")
    print(f"  live ball (dribbled):    {100 * stats['dribble'][graded].mean():.1f}% of graded frames")
    print(f"  same-team passes:        {stats['passes'][H] + stats['passes'][A]}")
    print(f"  throw-ins:               {stats['throw_ins']}, {stats['resolved']} with a last toucher found")
    print(f"  keeper catches:          {stats['catches']}  (a typical match has roughly 10-30)")
    for half, home_left in stats["ends"].items():
        print(f"  half {half}: home defends the {'left (x = 0)' if home_left else 'right'} end")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")

    v1.report(labels)
    v3.dead_ball_sources(labels, parts, signals)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v7.csv")


if __name__ == "__main__":
    main()
