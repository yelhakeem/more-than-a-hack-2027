"""v13: v12 plus set-piece set-ups and confirmed interceptions.

Changes from v12:
  - Set-piece set-up = dead ball. In live play the ball is rarely still for long, and when a
    player stops it the opponents close in. At a set piece the ball is placed and the
    opponents must back off (Laws of the Game: 9.15 m). Two tests, either is enough:
      * slow set-up: ball still for 3+ s and no opponent within 8 m, AND some sign of a set
        piece: opponents backed off at some point, the ball sits on a restart spot, or a
        stoppage trigger (ball out or whistle slowdown, v12) came in the 30 s before. An
        unpressed defender simply holding the ball still does not qualify;
      * quick free kick: ball still for 1+ s and at least two nearby opponents (within 15 m)
        moving away from it at more than 1 m/s on average.
    "Opponents" are the team NOT nearest to the ball. Keeper catches (v7) are excluded.
    A ball resting on a restart spot (goal area, corner, touchline, centre spot) adds a
    little extra weight. Confirmed set-ups are exempt from the 3 s minimum-stoppage cleanup,
    because quick free kicks can be shorter than that.
  - Confirmed interception: when control passes from one team to the other and the new team
    keeps the ball for at least 0.4 s or passes it to a teammate within 3 s, that team is
    backed firmly from the moment it gains control, so short possessions are not missed.

Usage:
    python solution/v13.py   # writes evaluation/results/v13.csv
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
from games import GAME, half_video_blocks  # noqa: E402
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, HOME, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402
import v11  # noqa: E402
import v12  # noqa: E402
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v13.csv")

STILL_MS = 0.5              # ball slower than this = at rest
SLOW_SETUP_S = 3.0
SLOW_SETUP_CLEAR_M = 8.0
QUICK_SETUP_S = 1.0
RETREAT_RADIUS_M = 15.0
RETREAT_MS = 1.0
RETREAT_MIN_PLAYERS = 2
RETREAT_SPAN_S = 0.5
SLOW_SETUP_TRIGGER_S = 30.0
SETUP_DEAD_CLUE = 2.5
RESTART_SPOT_BONUS = 0.5
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M = 5.5, 9.16
SPOT_TOLERANCE_M = 1.5
INTERCEPT_MIN_S = 0.4
INTERCEPT_PASS_S = 3.0
INTERCEPT_CLUE = 2.0


def runs_of(mask: np.ndarray) -> list:
    """[(start, stop)] of consecutive True frames."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def opponent_distances(t: dict, team_near: np.ndarray) -> np.ndarray:
    """Distance (m) from the ball to every player NOT on the nearest player's team (NaN otherwise)."""
    dist = np.hypot(t["px"] - t["bx"][:, None], t["py"] - t["by"][:, None])
    opp = (t["team"][None, :] != team_near[:, None]) & (team_near[:, None] >= 0)
    return np.where(opp, dist, np.nan)


def on_restart_spot(t: dict) -> np.ndarray:
    """Ball in a goal area, near a corner, on a touchline, or on the centre spot."""
    length, width = (float(v) for v in t["pitch"])
    x, y, tol = t["bx"], t["by"], SPOT_TOLERANCE_M
    goal_area = (np.abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH_M) & (
        (x <= GOAL_AREA_DEPTH_M) | (x >= length - GOAL_AREA_DEPTH_M))
    corner = (np.minimum(x, length - x) <= tol) & (np.minimum(y, width - y) <= tol)
    touchline = np.minimum(y, width - y) <= tol
    centre = np.hypot(x - length / 2, y - width / 2) <= tol
    return goal_area | corner | touchline | centre


def setup_clues(t: dict, signals: dict, speed: np.ndarray, keeper_live: np.ndarray) -> tuple:
    """Dead clue over set-piece set-ups; returns score, the set-up mask and counts."""
    n = len(speed)
    score = np.zeros((n, 3))
    still = (~np.isnan(t["bx"]) & ~signals["off"]
             & (np.nan_to_num(speed, nan=np.inf) < STILL_MS) & ~keeper_live)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        opp = opponent_distances(t, signals["team"])
        nearest_opp = np.nanmin(opp, axis=1)
        k = int(RETREAT_SPAN_S * FPS)
        moving_away = np.zeros(n)
        count_close = np.zeros(n)
        close = opp[:-k] <= RETREAT_RADIUS_M
        change = (opp[k:] - opp[:-k]) * FPS / k                 # m/s, positive = away from ball
        moving_away[:-k] = np.nanmean(np.where(close, change, np.nan), axis=1)
        count_close[:-k] = close.sum(axis=1)
    retreating = (np.nan_to_num(moving_away) > RETREAT_MS) & (count_close >= RETREAT_MIN_PLAYERS)
    spot = on_restart_spot(t)
    trig = np.flatnonzero(v12.triggers(t, signals))
    lookback = int(SLOW_SETUP_TRIGGER_S * FPS)
    window = int(SLOW_SETUP_S * FPS)

    setup = np.zeros(n, dtype=bool)
    slow = quick = 0
    for a, b in runs_of(still):
        length = b - a
        if length >= SLOW_SETUP_S * FPS and np.nanmedian(nearest_opp[a:b]) >= SLOW_SETUP_CLEAR_M:
            evidence = (retreating[max(0, a - window):b].any()
                        or spot[a:b].mean() > 0.5
                        or ((trig >= a - lookback) & (trig <= a)).any())
            if not evidence:
                continue                                         # an unpressed player holding the ball
            setup[a:b] = True
            slow += 1
        elif length >= QUICK_SETUP_S * FPS and retreating[a:b].any():
            setup[a:b] = True
            quick += 1
    near = v11.near_player_clue(signals)
    score[setup] -= near[setup]                                  # a placed ball is nobody's possession
    score[setup, D] += SETUP_DEAD_CLUE + RESTART_SPOT_BONUS * spot[setup]
    return score, setup, slow, quick


def interception_clues(signals: dict, speed: np.ndarray) -> tuple:
    """Firm clue for the new team over a confirmed change of control."""
    n = len(speed)
    score = np.zeros((n, 3))
    ctrl = v6.controls(signals["team"], signals["dist"], speed)
    confirmed = 0
    for i in range(1, len(ctrl)):
        (team_a, _, stop_a), (team_b, start_b, stop_b) = ctrl[i - 1], ctrl[i]
        if team_a == team_b or start_b - stop_a > INTERCEPT_PASS_S * FPS:
            continue
        kept = stop_b - start_b >= INTERCEPT_MIN_S * FPS
        passed_on = (i + 1 < len(ctrl) and ctrl[i + 1][0] == team_b
                     and ctrl[i + 1][1] - stop_b <= INTERCEPT_PASS_S * FPS)
        if kept or passed_on:
            state = H if team_b == HOME else A
            score[start_b:stop_b, state] += INTERCEPT_CLUE
            confirmed += 1
    return score, confirmed


def enforce_min_durations(labels: np.ndarray, protected: np.ndarray) -> tuple:
    """v5's cleanup, except stoppages overlapping a confirmed set-up are kept."""
    out = labels.copy()
    fixed = {"short stoppages": 0, "short plays": 0}
    for _, first, last in half_video_blocks(GAME):
        part = out[first:last + 1]
        prot = protected[first:last + 1]
        segments = v5.runs(part)
        for i in range(1, len(segments) - 1):
            label, a, b = segments[i]
            before, after = segments[i - 1][0], segments[i + 1][0]
            if (label == "D" and before == after and before in "HA"
                    and b - a < v5.MIN_DEAD_S * FPS and not prot[a:b].any()):
                part[a:b] = before
                fixed["short stoppages"] += 1
            elif label in "HA" and before == after == "D" and b - a < v5.MIN_PLAY_S * FPS:
                part[a:b] = "D"
                fixed["short plays"] += 1
    return out, fixed


def label_match(t: dict) -> tuple:
    _, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    speed = v6.ball_speed(t)
    tags = load_tags()
    runners = v8.runners_present(t)
    keeper_live = v7.keeper_clues(t, signals, speed)[0][:, D] < 0

    total = v10.build_v9_total(t, signals)
    total += v10.crossing_clues(t, signals, speed)[0]
    total += v12.backfill_clues(t, signals, tags)[0]
    total += v12.adjustment_clues(t, signals, speed, runners)[0]
    setup_score, setup, slow, quick = setup_clues(t, signals, speed, keeper_live)
    intercept, confirmed = interception_clues(signals, speed)
    labels, fixed = enforce_min_durations(v1.decode_halves(total + setup_score + intercept), setup)
    stats = dict(setup=setup, slow=slow, quick=quick, confirmed=confirmed, **fixed)
    return labels, stats


def main() -> None:
    tracking = load_tracking()
    labels, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))
    print("\nNEW CLUES USED")
    print(f"  set-ups, slow (3+ s, clear):     {stats['slow']}")
    print(f"  set-ups, quick (opponents back off): {stats['quick']}")
    print(f"  set-up frames:                   {100 * stats['setup'][graded].mean():.1f}% of graded frames")
    print(f"  confirmed interceptions:         {stats['confirmed']}")
    print(f"  blips removed:                   {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v13.csv")
    print("then:  python solution/diagnose.py evaluation/results/v13.csv")


if __name__ == "__main__":
    main()
