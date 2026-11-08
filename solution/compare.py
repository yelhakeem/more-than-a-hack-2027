"""Label-free comparison of two submissions: where they disagree, and what the data shows there.

Uses only the provided data (tracking, Smart Tagging) and our own output CSVs. Never the
answer key or the video.

Usage:
    python solution/compare.py v17 v19     # reads evaluation/results/v17.csv and v19.csv
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
from tracking import FPS, HOME, graded_mask, load_tracking, run_lengths  # noqa: E402
import v1  # noqa: E402
import v4  # noqa: E402
import v6  # noqa: E402
import v8  # noqa: E402
import v11  # noqa: E402

GOAL_WINDOW_S = 41.0          # v17's value (Augste & Cordes 2016)
STOP_BINS_S = (0, 3, 6, 12, 30, 60, 1e9)
OFF_BINS_S = (0, 0.5, 1, 2, 5, 1e9)


def read(name: str) -> np.ndarray:
    path = os.path.join(ROOT, "evaluation", "results", f"{name}.csv")
    with open(path) as handle:
        next(handle)
        return np.array([line.rstrip("\r\n").split(",")[1] for line in handle])


def runs(mask: np.ndarray) -> list:
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def evidence(t: dict, tags: list) -> dict:
    """Per-frame yes/no facts from the provided data."""
    n = len(t["bx"])
    _, signals = v1.clues(t)
    speed = v6.ball_speed(t)
    home_d, away_d = v11.nearest_by_team(t)
    seen = ~np.isnan(t["bx"])
    window = v4.set_piece_clues(n, tags)[0][:, v1.D] > 0
    goal = np.zeros(n, dtype=bool)
    for tag in tags:
        if v4.v2._norm(tag["tags"].get("Type")) == "kick off (after goal)":
            goal[max(0, tag["start_frame"] - int(GOAL_WINDOW_S * FPS)):tag["start_frame"]] = True
    return {
        "ball missing": ~seen,
        "ball off pitch (>1 m)": signals["off"],
        "ball fast on pitch (>5 m/s)": seen & ~signals["off"] & (np.nan_to_num(speed) > 5),
        "ball within 2 m of a player": seen & (signals["dist"] <= v1.CONTROL_RADIUS_M),
        "both teams within 2 m (duel)": (home_d <= 2) & (away_d <= 2),
        "set-piece tag window": window,
        "goal kick-off window": goal,
        "team average walking (<1.2 m/s)": signals["speed"] < v1.SLOW_SPEED_MS,
        "2+ players running": v8.runners_present(t),
    }


def share(mask: np.ndarray, where: np.ndarray) -> str:
    return f"{100 * mask[where].mean():5.1f}%" if where.any() else "    -"


def main() -> None:
    a_name, b_name = (sys.argv[1:3] if len(sys.argv) >= 3 else ("v17", "v19"))
    a, b = read(a_name), read(b_name)
    t = load_tracking()
    tags = load_tags()
    n = len(a)
    graded = graded_mask(n)
    ev = evidence(t, tags)

    print(f"COMPARING {a_name} vs {b_name} (label-free; graded frames only)")
    print(f"\n1. SAME LABEL on {100 * np.mean(a[graded] == b[graded]):.1f}% of graded frames")
    print(f"   rows = {a_name}, columns = {b_name}, % of graded frames")
    print("          " + "".join(f"{s:>8}" for s in "HAD"))
    for s in "HAD":
        print(f"   {s:<6}" + "".join(f"{100 * np.mean((a[graded] == s) & (b[graded] == r)):>8.1f}" for r in "HAD"))

    print("\n2. STOPPAGE LENGTHS (published mean 23.1 s; throw-ins 11.5 s)")
    head = "".join(f"{f'{lo:g}-{hi:g}s' if hi < 1e8 else f'{lo:g}s+':>9}" for lo, hi in zip(STOP_BINS_S, STOP_BINS_S[1:]))
    print(f"   {'':<6}{head}{'count':>8}{'median':>8}")
    for name, lab in ((a_name, a), (b_name, b)):
        lengths = []
        for _, first, last in half_video_blocks(GAME):
            lengths += list(run_lengths(lab[first:last + 1] == "D") / FPS)
        lengths = np.array(lengths)
        counts = [int(((lengths >= lo) & (lengths < hi)).sum()) for lo, hi in zip(STOP_BINS_S, STOP_BINS_S[1:])]
        print(f"   {name:<6}" + "".join(f"{c:>9}" for c in counts) + f"{len(lengths):>8}{np.median(lengths):>7.1f}s")

    a_dead, b_dead = graded & (a == "D"), graded & (b == "D")
    a_only, b_only = a_dead & ~b_dead, b_dead & ~a_dead
    both_live = graded & ~a_dead & ~b_dead
    print("\n3. WHAT THE DATA SHOWS WHERE THEY DISAGREE ON DEAD vs ALIVE")
    print(f"   {'':<34}{'all graded':>11}{a_name + ' dead only':>15}{b_name + ' dead only':>15}")
    print(f"   {'frames':<34}{int(graded.sum()):>11,}{int(a_only.sum()):>15,}{int(b_only.sum()):>15,}")
    for key, mask in ev.items():
        print(f"   {key:<34}{share(mask, graded):>11}{share(mask, a_only):>15}{share(mask, b_only):>15}")

    print("\n4. BALL SEEN OFF THE PITCH (>1 m): how long each time, and how often called dead")
    off_runs = [(s, e) for s, e in runs(ev["ball off pitch (>1 m)"] & graded)]
    print(f"   {'off for':<10}{'times':>7}{'frames':>9}{a_name + ' dead':>11}{b_name + ' dead':>11}"
          f"{'ball back on pitch within 1 s':>31}")
    seen = ~ev["ball missing"]
    for lo, hi in zip(OFF_BINS_S, OFF_BINS_S[1:]):
        sel = [(s, e) for s, e in off_runs if lo * FPS <= e - s < hi * FPS]
        if not sel:
            continue
        idx = np.concatenate([np.arange(s, e) for s, e in sel])
        back = np.mean([bool((seen[e:e + FPS] & ~ev["ball off pitch (>1 m)"][e:e + FPS]).any()) for _, e in sel])
        label = f"{lo:g}-{hi:g}s" if hi < 1e8 else f"{lo:g}s+"
        print(f"   {label:<10}{len(sel):>7}{len(idx):>9}{100 * np.mean(a[idx] == 'D'):>10.1f}%"
              f"{100 * np.mean(b[idx] == 'D'):>10.1f}%{100 * back:>30.1f}%")

    print("\n5. BOTH ALIVE BUT DIFFERENT TEAMS")
    diff = both_live & (a != b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        home_d, away_d = v11.nearest_by_team(t)
    clear = (np.minimum(home_d, away_d) <= 1.0) & (np.maximum(home_d, away_d) > 2.0) & ~ev["ball off pitch (>1 m)"]
    visible = np.where(home_d < away_d, "H", "A")
    vis = diff & clear
    print(f"   frames: {int(diff.sum()):,} ({100 * diff.sum() / graded.sum():.1f}% of graded); "
          f"possession clearly visible on {int(vis.sum()):,} of them")
    if vis.any():
        print(f"   matches the visible team there:  {a_name} {100 * np.mean(a[vis] == visible[vis]):.1f}%   "
              f"{b_name} {100 * np.mean(b[vis] == visible[vis]):.1f}%")
    dm = diff & ev["ball missing"]
    print(f"   ball missing on {100 * dm.sum() / max(diff.sum(), 1):.1f}% of them")


if __name__ == "__main__":
    main()
