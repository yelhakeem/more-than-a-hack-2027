"""Debugging aid: list where a submission disagrees with the PUBLIC 5-minute answer key.

For understanding which TYPE of mistake remains (pipeline debugging, allowed by RULES.md).
Never use it to pick numbers or choose between versions: the public window is 17 possessions.

Usage:
    python solution/diagnose.py evaluation/results/v9.csv

Prints one line per wrong stretch (>= 0.4 s) with context from the tracking, summarised as
percentages and averages only, plus a count of each type of mistake.
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
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, load_tracking  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v5  # noqa: E402
import v8  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
MIN_FRAMES = 10
TAG_WINDOW_S = 15


def read_labels(path: str) -> np.ndarray:
    with open(path) as handle:
        next(handle)
        return np.array([line.strip().split(",")[1].upper() for line in handle])


def kind_of(truth: str, pred: str) -> str:
    if truth == "D":
        return "missed stoppage"
    if pred == "D":
        return "false stoppage"
    return "wrong team"


def main() -> None:
    pred = read_labels(sys.argv[1])
    truth = read_labels(GT_PATH)
    scored = truth != "X"
    t = load_tracking()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = v1.mean_player_speed(t)
    runners = v8.runners_present(t)
    seen = ~np.isnan(t["bx"])
    tags = [(tg["start_frame"], v2._norm(tg["tags"].get("Type")) or v2._norm(tg["code"]))
            for tg in load_tags() if v2._norm(tg["code"]) in ("set pieces", "goals")]
    first = int(np.flatnonzero(scored)[0])

    wrong = scored & (pred != truth)
    print(f"{sys.argv[1]}: {int(wrong.sum())} of {int(scored.sum())} scored frames wrong "
          f"({100 * (1 - wrong.mean() / scored.mean()):.1f}% right)\n")
    print(f"  {'from':>6} {'secs':>5}  {'truth':>5} {'ours':>4}  {'type':<16} {'edge':<11} "
          f"{'ball seen':>9} {'speed':>6} {'runners':>7}  nearest tag")
    totals = {}
    for label, a, b in v5.runs(np.where(wrong, "W", "-")):
        if label != "W" or b - a < MIN_FRAMES:
            continue
        tr = max(set(truth[a:b]), key=list(truth[a:b]).count)
        pr = max(set(pred[a:b]), key=list(pred[a:b]).count)
        kind = kind_of(tr, pr)
        before, after = truth[a - 1], truth[b] if b < len(truth) else "X"
        edge = ("start" if before != tr else "") + ("+end" if after != tr else "")
        edge = edge.strip("+") or "middle"
        near = [(f - a, name) for f, name in tags if abs(f - a) <= TAG_WINDOW_S * FPS]
        near_text = (f"{near[0][1]} ({near[0][0] / FPS:+.0f} s)" if near else "-")
        print(f"  {(a - first) / FPS:6.1f} {(b - a) / FPS:5.1f}  {tr:>5} {pr:>4}  {kind:<16} {edge:<11} "
              f"{100 * seen[a:b].mean():8.0f}% {speed[a:b].mean():6.2f} {100 * runners[a:b].mean():6.0f}%  "
              f"{near_text}")
        totals[kind] = totals.get(kind, 0) + (b - a)
    print("\nWRONG FRAMES BY TYPE (stretches >= 0.4 s)")
    for kind, n in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<16} {n:6d} frames  ({n / FPS:5.1f} s)")
    print("\n'from' = seconds after the kick-off. 'edge' = whether the stretch touches the start or end")
    print("of a true possession or stoppage (timing errors) or sits in its middle (whole mistakes).")


if __name__ == "__main__":
    main()
