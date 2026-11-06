"""v1: possession as a hidden-state sequence (H, A, D) decoded with Viterbi.

Each frame gets a score per state from tracking clues; every state change pays a fee, so
possession only changes on sustained evidence. Settings come from football reasoning, not
from the public ground truth (see RULES.md).

Usage:
    python solution/v1.py   # writes evaluation/results/v1.csv and a label-free report
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
from tracking import FPS, HOME, load_tracking, run_lengths  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v1.csv")
STATES = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2

# Ball-to-player distance clues
CONTROL_RADIUS_M = 2.0
NEAR_RADIUS_M = 5.0
STRONG_CLUE = 2.0
WEAK_CLUE = 0.5
# Ball out of play
OFF_PITCH_MARGIN_M = 1.0    # ignore line-hugging tracking noise
OFF_PITCH_CLUE = 3.0
# Collective player speed (1 s average)
SLOW_SPEED_MS = 1.2
SLOW_CLUE = 1.0             # scaled by how far below the threshold, capped at 2x
FAST_SPEED_MS = 2.5
FAST_CLUE = 0.5
# State-change fees
SWITCH_FEE = 30.0           # H <-> A: ~0.6 s of close control
DEAD_FEE = 12.0             # into or out of D
# Graded from 110,808 but the second half kicks off here; everything before is D (FAQ.md)
SECOND_HALF_KICKOFF = 113742


def nearest_player(t: dict) -> tuple:
    """Per frame: distance (m) to the nearest tracked player and that player's team.

    Distance is NaN and team -1 where the ball or all players are missing.
    """
    dist = np.hypot(t["px"] - t["bx"][:, None], t["py"] - t["by"][:, None])
    valid = ~np.all(np.isnan(dist), axis=1)
    idx = np.nanargmin(dist[valid], axis=1)

    best = np.full(len(dist), np.nan, dtype=np.float32)
    team = np.full(len(dist), -1, dtype=np.int8)
    best[valid] = dist[valid, idx]
    team[valid] = t["team"][idx]
    return best, team


def mean_player_speed(t: dict) -> np.ndarray:
    """Average speed of tracked players, smoothed over 1 s. Frames with no players get the midpoint."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN frames
        speed = np.nanmean(t["ps"], axis=1)
    speed = np.nan_to_num(speed, nan=(SLOW_SPEED_MS + FAST_SPEED_MS) / 2)
    return np.convolve(speed, np.ones(FPS) / FPS, mode="same")


def clues(t: dict) -> tuple:
    """Per-frame scores for (H, A, D), shape (frames, 3), plus the intermediate signals."""
    score = np.zeros((len(t["bx"]), 3))

    dist, team = nearest_player(t)
    strong = dist <= CONTROL_RADIUS_M
    weak = (dist > CONTROL_RADIUS_M) & (dist <= NEAR_RADIUS_M)
    for state, side in ((H, HOME), (A, 1 - HOME)):
        score[strong & (team == side), state] += STRONG_CLUE
        score[weak & (team == side), state] += WEAK_CLUE

    length, width = (float(v) for v in t["pitch"])
    m = OFF_PITCH_MARGIN_M
    off = (t["bx"] < -m) | (t["bx"] > length + m) | (t["by"] < -m) | (t["by"] > width + m)
    score[off, D] += OFF_PITCH_CLUE

    speed = mean_player_speed(t)
    slow = speed < SLOW_SPEED_MS
    score[slow, D] += SLOW_CLUE * np.minimum((SLOW_SPEED_MS - speed[slow]) / SLOW_SPEED_MS * 2, 2)
    score[speed > FAST_SPEED_MS, D] -= FAST_CLUE
    return score, dict(dist=dist, off=off, speed=speed)


def best_sequence(score: np.ndarray) -> np.ndarray:
    """Viterbi decode: highest total score minus state-change fees."""
    fee = np.array([[0, SWITCH_FEE, DEAD_FEE],
                    [SWITCH_FEE, 0, DEAD_FEE],
                    [DEAD_FEE, DEAD_FEE, 0]], dtype=np.float64)
    n = len(score)
    total = score[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    cols = np.arange(3)
    for i in range(1, n):
        options = total[:, None] - fee          # [from, to]
        back[i] = np.argmax(options, axis=0)
        total = options[back[i], cols] + score[i]

    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode_halves(score: np.ndarray) -> np.ndarray:
    """Decode each half separately; X outside the halves, D before the second-half kick-off."""
    labels = np.full(len(score), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        start = first
        if half == 2 and first < SECOND_HALF_KICKOFF:
            labels[first:SECOND_HALF_KICKOFF] = "D"
            start = SECOND_HALF_KICKOFF
        labels[start:last + 1] = STATES[best_sequence(score[start:last + 1])]
    return labels


def label_match(t: dict) -> np.ndarray:
    return decode_halves(clues(t)[0])


def write_submission(path: str, labels: np.ndarray) -> None:
    """Write the frame,label CSV the scorer expects."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{label}\n" for i, label in enumerate(labels))


def report(labels: np.ndarray) -> None:
    """Label-free check per half: possession count, lengths and state shares."""
    print("\nREPORT (no answer key used)")
    print(f"  {'half':<6} {'poss':>8} {'median s':>9} {'longest s':>9} "
          f"{'H %':>7} {'A %':>7} {'D %':>7}")
    for half, first, last in half_video_blocks(GAME):
        part = labels[first:last + 1]
        poss = np.concatenate([run_lengths(part == s) for s in "HA"])
        if poss.size == 0:
            poss = np.array([0])
        h, a, d = (100.0 * np.mean(part == s) for s in "HAD")
        print(f"  {half:<6} {len(poss):>8} {np.median(poss) / FPS:>9.1f} "
              f"{poss.max() / FPS:>9.1f} {h:>7.1f} {a:>7.1f} {d:>7.1f}")


def main() -> None:
    labels = label_match(load_tracking())
    write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v1.csv")
    print("then:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
