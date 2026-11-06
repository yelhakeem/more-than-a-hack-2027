"""v9: v8 plus player-action cues for ball-missing stretches.

Changes from v8:
  - Short gaps (<= 1 s): the ball's position is filled in on a straight line between the
    last and next sighting, so the "ball near a player" clue keeps working (half strength).
  - Team-shape cues, used only while the ball is missing:
      * pushing forward: the team in possession moves its average position towards the
        opponents' goal; the team without it drops back;
      * spread: the team in possession stretches wide and long, the defenders squeeze.
  - Self-check without the answer key: each cue is tested on frames where possession is
    visible (ball within 1 m of a player). Its weight comes from how often it agrees:
    50% agreement (a coin flip) gives weight 0, 75% or more gives the full weight.

Loose or contested balls keep the last team's possession, as CHALLENGE.md defines: with no
clue strong enough to pay the change fee, the decoder carries the previous state on.

Usage:
    python solution/v9.py   # writes evaluation/results/v9.csv
"""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
from tracking import FPS, HOME, graded_mask, load_tracking, run_lengths  # noqa: E402
import v1  # noqa: E402
import v3  # noqa: E402
import v5  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v9.csv")

FILL_MAX_S = 1.0            # fill ball gaps up to this long
FILL_CLUE_SCALE = 0.5       # filled-in positions count half as much as seen ones
SHAPE_SMOOTH_S = 2.0        # smoothing for the team-shape signals
VISIBLE_RADIUS_M = 1.0      # "possession visible" for the self-check
SHAPE_MAX_CLUE = 0.5        # per frame, at full weight
SHAPE_CLIP = 2.0            # cue values are capped at 2 standard deviations


def fill_short_gaps(t: dict) -> tuple:
    """Copy of the tracking with ball gaps <= FILL_MAX_S filled linearly, plus a 'filled' mask."""
    bx, by = t["bx"].astype(float).copy(), t["by"].astype(float).copy()
    seen = ~np.isnan(bx)
    filled = np.zeros(len(bx), dtype=bool)
    padded = np.concatenate(([0], (~seen).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    for start, stop in zip(edges[::2], edges[1::2]):
        if start == 0 or stop >= len(bx) or stop - start > FILL_MAX_S * FPS:
            continue
        w = np.arange(1, stop - start + 1) / (stop - start + 1)
        bx[start:stop] = bx[start - 1] + w * (bx[stop] - bx[start - 1])
        by[start:stop] = by[start - 1] + w * (by[stop] - by[start - 1])
        filled[start:stop] = True
    t_filled = dict(t, bx=bx.astype(np.float32), by=by.astype(np.float32))
    return t_filled, filled


def fill_clues(t: dict, filled: np.ndarray) -> np.ndarray:
    """v1's ball-near-player clue on filled frames only, at half strength."""
    score = np.zeros((len(filled), 3))
    dist, team = v1.nearest_player(t)
    for state, side in ((H, HOME), (A, 1 - HOME)):
        mine = filled & (team == side)
        score[mine & (dist <= v1.CONTROL_RADIUS_M), state] += v1.STRONG_CLUE * FILL_CLUE_SCALE
        near = (dist > v1.CONTROL_RADIUS_M) & (dist <= v1.NEAR_RADIUS_M)
        score[mine & near, state] += v1.WEAK_CLUE * FILL_CLUE_SCALE
    return score


def smooth(x: np.ndarray) -> np.ndarray:
    k = int(SHAPE_SMOOTH_S * FPS)
    return np.convolve(np.nan_to_num(x), np.ones(k) / k, mode="same")


def shape_signals(t: dict) -> dict:
    """Per-frame cues, positive = favours home: forward push difference and spread ratio."""
    n = len(t["bx"])
    home = t["team"] == HOME
    ends = v7.defended_ends(t)
    attack_sign = np.zeros(n)           # +1 if home attacks towards larger x
    for half, first, last in half_video_blocks(GAME):
        attack_sign[first:last + 1] = 1.0 if ends[half] else -1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cx_home = np.nanmean(t["px"][:, home], axis=1)
        cx_away = np.nanmean(t["px"][:, ~home], axis=1)
        spread_home = np.nanstd(t["px"][:, home], axis=1) * np.nanstd(t["py"][:, home], axis=1)
        spread_away = np.nanstd(t["px"][:, ~home], axis=1) * np.nanstd(t["py"][:, ~home], axis=1)

    k = v8.v6.SPEED_SPAN * 5
    def velocity(x):
        v = np.zeros(n)
        v[k:-k] = (x[2 * k:] - x[:-2 * k]) * FPS / (2 * k)
        return v
    # home pushing forward and away dropping back both favour home
    push = smooth(velocity(np.nan_to_num(cx_home)) * attack_sign
                  - velocity(np.nan_to_num(cx_away)) * -attack_sign)
    with np.errstate(divide="ignore", invalid="ignore"):
        spread = smooth(np.log(spread_home / spread_away))
    return {"pushing forward": push, "spread out": spread}


def shape_clues(t: dict, signals: dict, missing: np.ndarray) -> tuple:
    """Team clues from shape cues on ball-missing frames, weighted by self-check agreement."""
    n = len(t["bx"])
    score = np.zeros((n, 3))
    graded = graded_mask(n)
    visible = graded & (signals["dist"] <= VISIBLE_RADIUS_M) & ~signals["off"]
    visible_home = signals["team"][visible] == HOME
    results = {}
    for name, cue in shape_signals(t).items():
        sd = np.std(cue[graded]) or 1.0
        z = np.clip(cue / sd, -SHAPE_CLIP, SHAPE_CLIP) / SHAPE_CLIP     # -1..1
        decided = z[visible] != 0
        agree = float(np.mean((z[visible][decided] > 0) == visible_home[decided])) if decided.any() else 0.5
        weight = float(np.clip((agree - 0.5) / 0.25, 0, 1)) * SHAPE_MAX_CLUE
        score[missing, H] += weight * np.maximum(z[missing], 0)
        score[missing, A] += weight * np.maximum(-z[missing], 0)
        results[name] = (agree, weight)
    return score, results


def label_match(t: dict) -> tuple:
    labels, parts, signals, stats = v8.label_match(t)          # v8 builds its clues...
    # ...but v9 needs the total score itself, so rebuild it the same way and add the new cues.
    total = v8_total(t, signals)
    t_filled, filled = fill_short_gaps(t)
    total += fill_clues(t_filled, filled)
    missing = np.isnan(t["bx"]) & ~filled
    shape, results = shape_clues(t, signals, missing)
    total += shape
    labels, fixed = v5.enforce_min_durations(v1.decode_halves(total))
    stats.update(fixed, filled=filled, missing=missing, shape=results)
    return labels, parts, signals, stats


def v8_total(t: dict, signals: dict) -> np.ndarray:
    """v8's total clue score (same parts, same order as v8.label_match)."""
    base, _ = v1.clues(t)
    n = len(base)
    speed = v8.v6.ball_speed(t)
    tags = v8.load_tags()
    restarts = v7.restart_frames(tags)
    runners = v8.runners_present(t)
    base[runners, D] -= v8.walking_clue(signals["speed"])[runners]
    total = (base + v8.v4.set_piece_clues(n, tags)[0] + v3.goal_clues(n, tags)[0]
             + v3.vanish_clues(t, signals["speed"])[0]
             + v8.live_ball_clues(t, signals, speed, restarts, runners)[0]
             + v8.v6.pass_clues(t, signals, speed)[0]
             + v7.throw_in_clues(t, signals, restarts)[0]
             + v7.keeper_clues(t, signals, speed)[0]
             + v8.ball_out_clues(t, signals, tags)[0]
             + v8.shot_clues(t, signals, speed)[0])
    return total


def main() -> None:
    tracking = load_tracking()
    labels, parts, signals, stats = label_match(tracking)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))

    print("\nBALL-MISSING STRETCHES")
    print(f"  short gaps filled in:    {100 * stats['filled'][graded].mean():.1f}% of graded frames")
    print(f"  still missing:           {100 * stats['missing'][graded].mean():.1f}% of graded frames")
    print("\nSHAPE CUES, CHECKED ON FRAMES WHERE POSSESSION IS VISIBLE (no answer key)")
    for name, (agree, weight) in stats["shape"].items():
        print(f"  {name:<16} agrees {100 * agree:5.1f}% of the time  ->  weight {weight:.2f} "
              f"(max {SHAPE_MAX_CLUE})")
    print(f"\n  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v9.csv")


if __name__ == "__main__":
    main()
