"""Version 1: possession as a chain of states, not a pile of separate frames.

The idea, in plain words
------------------------
At every frame the match is in one of three states: H (home has the ball), A (away has it) or
D (dead ball). We collect clues for each frame, then pick the single sequence of states that
best fits all the clues, while charging a "fee" every time the state changes. Because changing
state is expensive, one stray clue can't flip possession; it takes a run of clues. That gives
whole, clean possessions by design -- which is exactly what the ranking metric (spellF1) rewards.
The best sequence is found with the Viterbi algorithm, a standard method for this kind of
problem.

The clues used in version 1
---------------------------
  * Ball close to a player: that player's team probably has it. Very close (<= 2 m) is a strong
    clue, close (2-5 m) a weak one, farther away no clue at all.
  * Ball off the pitch: probably a dead ball.
  * Everyone moving slowly: probably a stoppage (players walk while the ball is out, a foul is
    given, or a goal is celebrated).
  * Before the second-half kick-off: always dead (a published fact, see FAQ.md).
  * Ball not seen: no ball clue at all, so the current state simply carries on.

Every number below was chosen from football common sense, NOT by checking the score on the
5 public minutes -- the rules forbid that, and EVALUATION.md shows it picks the wrong answer.

Run it from the project folder:

    python solution/v1.py

It writes evaluation/results/v1.csv and prints a report that uses no answer key.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
from tracking import HOME, load_tracking  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v1.csv")
FPS = 25
STATES = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2

# ---- Settings, all from football reasoning ---------------------------------------------------
CONTROL_RADIUS_M = 2.0      # ball this close to a player = that player is playing it
NEAR_RADIUS_M = 5.0         # ball this close = weak hint
STRONG_CLUE = 2.0           # how much a "very close" frame favours that team
WEAK_CLUE = 0.5             # how much a "close" frame favours that team
OFF_PITCH_MARGIN_M = 1.0    # ball must be this far past a line, so line-hugging noise is ignored
OFF_PITCH_CLUE = 3.0        # how much a clearly-out ball favours D
SLOW_SPEED_MS = 1.2         # average player speed below this = everyone is walking
SLOW_CLUE = 1.0             # how much "everyone walking" favours D (scaled by how slow)
FAST_SPEED_MS = 2.5         # average speed above this = play is clearly live
FAST_CLUE = 0.5             # how much "everyone running" counts against D
SWITCH_FEE = 30.0           # cost of H <-> A: needs ~15 frames (0.6 s) of close control to pay
DEAD_FEE = 12.0             # cost of entering or leaving D
SECOND_HALF_KICKOFF = 113742  # FAQ.md: graded from 110,808 but kicks off here; before = all D


def nearest_player(t):
    """Per frame: distance (m) from ball to nearest tracked player, and that player's team.

    NaN distance / team -1 where the ball or every player is missing.
    """
    dx = t["px"] - t["bx"][:, None]
    dy = t["py"] - t["by"][:, None]
    dist = np.sqrt(dx * dx + dy * dy)
    has_any = ~np.all(np.isnan(dist), axis=1)
    nearest = np.zeros(len(dist), dtype=np.int64)
    nearest[has_any] = np.nanargmin(dist[has_any], axis=1)
    best = np.full(len(dist), np.nan, dtype=np.float32)
    best[has_any] = dist[has_any, nearest[has_any]]
    team = np.full(len(dist), -1, dtype=np.int8)
    team[has_any] = t["team"][nearest[has_any]]
    return best, team


def clues(t):
    """A score per frame for each state (higher = more likely). Shape (frames, 3)."""
    n = len(t["bx"])
    score = np.zeros((n, 3), dtype=np.float64)

    # 1. Ball close to a player -> that team.
    dist, team = nearest_player(t)
    strong = dist <= CONTROL_RADIUS_M
    weak = (dist > CONTROL_RADIUS_M) & (dist <= NEAR_RADIUS_M)
    for side in (H, A):
        is_side = team == (HOME if side == H else 1 - HOME)
        score[strong & is_side, side] += STRONG_CLUE
        score[weak & is_side, side] += WEAK_CLUE

    # 2. Ball clearly off the pitch -> dead.
    length, width = [float(v) for v in t["pitch"]]
    m = OFF_PITCH_MARGIN_M
    off = ((t["bx"] < -m) | (t["bx"] > length + m) | (t["by"] < -m) | (t["by"] > width + m))
    score[off, D] += OFF_PITCH_CLUE

    # 3. Everyone slow -> stoppage; everyone fast -> live. Uses players, so it works even when
    #    the ball is missing. Averaged over 1 s so a single noisy frame doesn't matter.
    with np.errstate(all="ignore"):
        speed = np.nanmean(t["ps"], axis=1)
    speed = np.nan_to_num(speed, nan=(SLOW_SPEED_MS + FAST_SPEED_MS) / 2)
    speed = np.convolve(speed, np.ones(FPS) / FPS, mode="same")
    slow = speed < SLOW_SPEED_MS
    score[slow, D] += SLOW_CLUE * np.minimum((SLOW_SPEED_MS - speed[slow]) / SLOW_SPEED_MS * 2, 2)
    score[speed > FAST_SPEED_MS, D] -= FAST_CLUE
    return score, dict(dist=dist, off=off, speed=speed)


def best_sequence(score):
    """Viterbi: the state sequence with the highest total clue score minus change fees."""
    n = len(score)
    fee = np.array([[0, SWITCH_FEE, DEAD_FEE],
                    [SWITCH_FEE, 0, DEAD_FEE],
                    [DEAD_FEE, DEAD_FEE, 0]], dtype=np.float64)
    total = score[0].copy()
    came_from = np.zeros((n, 3), dtype=np.int8)
    for i in range(1, n):
        options = total[:, None] - fee            # options[from, to]
        came_from[i] = np.argmax(options, axis=0)
        total = options[came_from[i], [0, 1, 2]] + score[i]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for i in range(n - 1, 0, -1):
        path[i - 1] = came_from[i, path[i]]
    return path


def label_match(t):
    """Labels for every video frame: H/A/D inside the halves, X outside."""
    n = len(t["bx"])
    score, _ = clues(t)
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        start = first
        if half == 2 and first < SECOND_HALF_KICKOFF:
            labels[first:SECOND_HALF_KICKOFF] = "D"
            start = SECOND_HALF_KICKOFF
        labels[start:last + 1] = STATES[best_sequence(score[start:last + 1])]
    return labels


def write_submission(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def report(labels):
    """Checks that use no answer key: possession counts and lengths, per half."""
    print("\nREPORT (no answer key used)")
    print("  {:<6} {:>8} {:>9} {:>9} {:>7} {:>7} {:>7}".format(
        "half", "poss", "median s", "longest s", "H %", "A %", "D %"))
    for half, first, last in half_video_blocks(GAME):
        part = labels[first:last + 1]
        runs, current, length = [], part[0], 0
        for label in part:
            if label == current:
                length += 1
            else:
                runs.append((current, length))
                current, length = label, 1
        runs.append((current, length))
        poss = [n for label, n in runs if label in "HA"]
        poss = np.array(poss) if poss else np.array([0])
        share = {s: 100.0 * np.mean(part == s) for s in "HAD"}
        print("  {:<6} {:>8} {:>9.1f} {:>9.1f} {:>7.1f} {:>7.1f} {:>7.1f}".format(
            half, len(poss), np.median(poss) / FPS, poss.max() / FPS,
            share["H"], share["A"], share["D"]))


if __name__ == "__main__":
    tracking = load_tracking()
    labels = label_match(tracking)
    write_submission(OUT_PATH, labels)
    print("wrote {}".format(os.path.relpath(OUT_PATH, ROOT)))
    report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v1.csv")
    print("then:  python evaluation/run_evaluation.py")
