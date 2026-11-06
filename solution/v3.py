"""v3: v2 with fewer false stoppages and a working goal clue.

Changes from v2:
  - Vanish clue: the ball must vanish near a line AND players must slow down within 5 s.
    In live wide play the ball is often lost near a touchline while play continues.
    Capped at 15 s (was 30 s).
  - Goal clue: dead for the 45 s before each "Kick Off (After Goal)" tag. v2 tried to pair
    each goal with its kick-off and paired none.
  - Report: how many dead-ball frames each clue supports, so we can see which clue drives
    the stoppages without using the answer key.

Usage:
    python solution/v3.py   # writes evaluation/results/v3.csv
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
from tracking import FPS, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
from v1 import D  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v3.csv")

# Ball vanished near a line, confirmed by players slowing down
SLOWDOWN_WINDOW_S = 5.0     # players must drop below the walking threshold within this time
VANISH_MAX_S = 15.0
VANISH_CLUE = 1.0
# Goals: the restart after a goal always comes well after it
BEFORE_GOAL_KICKOFF_S = 45.0
GOAL_DEAD_CLUE = 2.5


def goal_clues(n_frames: int, tags: list) -> tuple:
    """Dead clue for the stretch just before every "Kick Off (After Goal)"."""
    score = np.zeros((n_frames, 3))
    kickoffs = [t["start_frame"] for t in tags
                if v2._norm(t["tags"].get("Type")) == "kick off (after goal)"]
    for kickoff in kickoffs:
        score[max(0, kickoff - int(BEFORE_GOAL_KICKOFF_S * FPS)):kickoff, D] += GOAL_DEAD_CLUE
    return score, len(kickoffs)


def vanish_clues(t: dict, speed: np.ndarray) -> tuple:
    """Dead clue over ball-missing stretches that start near a line and coincide with a slowdown."""
    n = len(t["bx"])
    score = np.zeros((n, 3))
    length, width = (float(v) for v in t["pitch"])
    m = v2.LINE_MARGIN_M
    seen = ~np.isnan(t["bx"])
    near_line = seen & ((t["bx"] < m) | (t["bx"] > length - m) | (t["by"] < m) | (t["by"] > width - m))

    padded = np.concatenate(([0], (~seen).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    candidates = confirmed = 0
    for start, stop in zip(edges[::2], edges[1::2]):
        if start == 0 or not near_line[start - 1] or stop - start < v2.VANISH_MIN_S * FPS:
            continue
        candidates += 1
        window = speed[start:min(stop, start + int(SLOWDOWN_WINDOW_S * FPS))]
        if window.size and window.min() < v1.SLOW_SPEED_MS:
            score[start:min(stop, start + int(VANISH_MAX_S * FPS)), D] += VANISH_CLUE
            confirmed += 1
    return score, candidates, confirmed


def label_match(t: dict) -> tuple:
    """Labels for every frame, plus the clue scores for the report."""
    base, signals = v1.clues(t)
    n = len(base)
    tags = load_tags()
    goal_score, goal_kickoffs = goal_clues(n, tags)
    vanish, candidates, confirmed = vanish_clues(t, signals["speed"])
    parts = {
        "set pieces": v2.set_piece_clues(n, tags)[0],
        "goal restarts": goal_score,
        "ball vanished": vanish,
    }
    labels = v1.decode_halves(base + sum(parts.values()))
    stats = dict(candidates=candidates, confirmed=confirmed, goal_kickoffs=goal_kickoffs)
    return labels, parts, signals, stats


def dead_ball_sources(labels: np.ndarray, parts: dict, signals: dict) -> None:
    """Share of graded D frames that each clue supports (frames can count for several)."""
    graded = graded_mask(len(labels))
    dead = graded & (labels == "D")
    total = max(int(dead.sum()), 1)
    sources = {name: score[:, D] > 0 for name, score in parts.items()}
    sources["ball off pitch"] = signals["off"]
    sources["players walking"] = signals["speed"] < v1.SLOW_SPEED_MS
    second_half_start = half_video_blocks(GAME)[1][1]
    pre_kickoff = np.zeros(len(labels), dtype=bool)
    pre_kickoff[second_half_start:v1.SECOND_HALF_KICKOFF] = True
    sources["before 2nd-half kick-off"] = pre_kickoff
    any_clue = np.zeros(len(labels), dtype=bool)

    print("\nWHERE THE DEAD-BALL CALLS COME FROM (graded D frames; one frame can have several)")
    for name, active in sources.items():
        any_clue |= active
        print(f"  {name:<26} {100 * (dead & active).sum() / total:5.1f}%")
    print(f"  {'no dead-ball clue at all':<26} {100 * (dead & ~any_clue).sum() / total:5.1f}%"
          "  (carried over from the previous state)")


def main() -> None:
    labels, parts, signals, stats = label_match(load_tracking())
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print("\nNEW CLUES USED")
    print(f"  ball vanished near line: {stats['candidates']} stretches, "
          f"{stats['confirmed']} confirmed by a slowdown")
    print(f"  goal kick-offs used:     {stats['goal_kickoffs']}")
    v1.report(labels)
    dead_ball_sources(labels, parts, signals)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v3.csv")
    print("then:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
