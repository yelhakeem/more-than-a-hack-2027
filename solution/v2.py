"""v2: v1 plus dead-ball and restart clues.

Adds three clues to v1's state model, targeting stoppages where the ball leaves the tracking:
  - Smart Tagging set pieces: dead before the restart, restarting team just after it
    (throw-ins are tagged Team=N/A, so they only add the dead part).
  - Goals: dead from a GOALS tag to the next "Kick Off (After Goal)".
  - Ball vanished near a line: dead while missing, if last seen within 2 m of a line.

Smart Tagging is automatic, so its clues are moderate and can be outvoted by the tracking.

Usage:
    python solution/v2.py   # writes evaluation/results/v2.csv, a report and speed percentiles
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
from tracking import FPS, graded_mask, load_tracking, run_lengths  # noqa: E402
import v1  # noqa: E402
from v1 import A, D, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v2.csv")

# Set pieces (tags mark a ~5 s window around the restart)
SET_PIECE_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)")
BEFORE_TAG_S = 4.0          # dead clue starts this long before the window
SET_PIECE_DEAD_CLUE = 1.5   # ... and runs to the window's midpoint
AFTER_TAG_S = 3.0           # restarting-team clue runs from the midpoint to this after the end
RESTART_TEAM_CLUE = 1.0
# Goals
GOAL_DEAD_CLUE = 2.5
GOAL_TO_KICKOFF_MAX_S = 180
# Ball vanished near a line
LINE_MARGIN_M = 2.0
VANISH_MIN_S = 2.0
VANISH_MAX_S = 30.0
VANISH_CLUE = 1.0


def _norm(text) -> str:
    """Case-insensitive matching: tag codes are upper case, values title case."""
    return (text or "").strip().lower()


def team_side(tag_value):
    """Map a Smart Tagging club name to H or A; None for N/A."""
    value = _norm(tag_value)
    if "barcelona" in value:
        return H
    if "atletic" in value or "athletic" in value:
        return A
    return None


def set_piece_clues(n_frames: int) -> tuple:
    """Scores from set pieces and goals, plus counts of what was used."""
    score = np.zeros((n_frames, 3))
    counts = {"set pieces": 0, "with team": 0, "goal stoppages": 0}
    tags = load_tags()

    for tag in tags:
        if _norm(tag["code"]) != "set pieces" or _norm(tag["tags"].get("Type")) not in SET_PIECE_TYPES:
            continue
        start, end = tag["start_frame"], tag["end_frame"]
        middle = (start + end) // 2
        score[max(0, start - int(BEFORE_TAG_S * FPS)):middle, D] += SET_PIECE_DEAD_CLUE
        counts["set pieces"] += 1

        side = team_side(tag["tags"].get("Team"))
        if side is not None:
            score[middle:min(n_frames, end + int(AFTER_TAG_S * FPS)), side] += RESTART_TEAM_CLUE
            counts["with team"] += 1

    goals = sorted(t["end_frame"] for t in tags if _norm(t["code"]) == "goals")
    kickoffs = sorted(t["start_frame"] for t in tags
                      if _norm(t["tags"].get("Type")) == "kick off (after goal)")
    for goal in goals:
        later = [k for k in kickoffs if goal < k <= goal + GOAL_TO_KICKOFF_MAX_S * FPS]
        if later:
            score[goal:later[0], D] += GOAL_DEAD_CLUE
            counts["goal stoppages"] += 1
    return score, counts


def vanish_clues(t: dict) -> tuple:
    """Dead clue over ball-missing stretches that start right after the ball was near a line."""
    n = len(t["bx"])
    score = np.zeros((n, 3))
    length, width = (float(v) for v in t["pitch"])
    m = LINE_MARGIN_M
    seen = ~np.isnan(t["bx"])
    near_line = seen & ((t["bx"] < m) | (t["bx"] > length - m) | (t["by"] < m) | (t["by"] > width - m))

    padded = np.concatenate(([0], (~seen).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    used = 0
    for start, stop in zip(edges[::2], edges[1::2]):
        if start > 0 and near_line[start - 1] and stop - start >= VANISH_MIN_S * FPS:
            score[start:min(stop, start + int(VANISH_MAX_S * FPS)), D] += VANISH_CLUE
            used += 1
    return score, used


def label_match(t: dict) -> tuple:
    """Labels for every frame, plus diagnostics for the report."""
    score, signals = v1.clues(t)
    sp_score, sp_counts = set_piece_clues(len(score))
    vn_score, vanished = vanish_clues(t)
    labels = v1.decode_halves(score + sp_score + vn_score)
    return labels, sp_counts, vanished, signals["speed"]


def speed_report(speed: np.ndarray) -> None:
    """Percentiles of average player speed on graded frames (label-free)."""
    graded = graded_mask(len(speed))
    pct = np.percentile(speed[graded], [10, 25, 50, 75, 90])
    print("\nPLAYER SPEED (average over tracked players, 1 s smoothing, graded frames)")
    print("  10% / 25% / 50% / 75% / 90%:  " + " / ".join(f"{p:.2f}" for p in pct) + " m/s")
    print(f"  share below the {v1.SLOW_SPEED_MS} m/s 'walking' threshold:  "
          f"{100 * np.mean(speed[graded] < v1.SLOW_SPEED_MS):.1f}%")


def main() -> None:
    labels, sp_counts, vanished, speed = label_match(load_tracking())
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print("\nNEW CLUES USED")
    print(f"  set pieces used:         {sp_counts['set pieces']} ({sp_counts['with team']} with a team)")
    print(f"  goal-to-kick-off gaps:   {sp_counts['goal stoppages']}")
    print(f"  ball vanished near line: {vanished} stretches")
    v1.report(labels)
    speed_report(speed)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v2.csv")
    print("then:  python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
