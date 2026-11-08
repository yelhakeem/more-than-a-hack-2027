"""Label-free check: where is the real restart inside each Smart Tagging set-piece window?

v4 places every restart at 68% of the tag window (measured on the two kick-offs). restarts.py
looks for the restart the Laws describe (throw-in: ball back over the touchline; corner, goal
kick, free kick, kick-off: a still ball struck from the right spot). This prints how far the
detected restart is from v4's estimate, per set-piece type.

Uses only the tracking, the tags and our own output CSV. Never the answer key or the video.

Usage:
    python solution/setpiece_check.py v20
"""

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, load_tracking  # noqa: E402
import compare  # noqa: E402
import restarts  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v6  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "v20"
    labels = compare.read(name)
    t = load_tracking()
    tags = load_tags()
    _, signals = v1.clues(t)
    speed = v6.ball_speed(t)
    found = restarts.find_restarts(t, speed, signals["off"], tags, v2._norm)

    by_type = defaultdict(list)
    for tag in tags:
        if id(tag) not in found:
            continue
        start, end = tag["start_frame"], tag["end_frame"]
        estimate = start + int(round(restarts.RESTART_FRACTION * (end - start)))
        frame, detected = found[id(tag)]
        lo, hi = sorted((frame, estimate))
        dead = float(np.mean(labels[lo:hi] == "D")) if hi > lo else np.nan
        by_type[v2._norm(tag["tags"].get("Type"))].append(
            ((frame - estimate) / FPS if detected else np.nan, (end - start) / FPS, hi - lo if detected else 0, dead))

    print(f"DETECTED RESTART vs v4's ESTIMATE (label-free; labels from {name}.csv)")
    print(f"  {'type':<22}{'tags':>5}{'found':>6}{'tag length':>11}{'detected - estimate (s)':>32}"
          f"{'>2 s away':>10}{'frames between':>15}{'now dead':>9}")
    total = 0
    for kind, rows in sorted(by_type.items()):
        off = np.array([r[0] for r in rows])
        ok = ~np.isnan(off)
        frames = int(sum(r[2] for r in rows))
        total += frames
        dead = [r[3] for r in rows if r[2] > 0 and not np.isnan(r[3])]
        spread = (f"{np.median(off[ok]):+.1f} (p25 {np.percentile(off[ok], 25):+.1f}, "
                  f"p75 {np.percentile(off[ok], 75):+.1f})") if ok.any() else "-"
        print(f"  {kind:<22}{len(rows):>5}{int(ok.sum()):>6}{np.median([r[1] for r in rows]):>10.1f}s"
              f"{spread:>32}{int((np.abs(off[ok]) > 2).sum()):>10}{frames:>15}"
              f"{(f'{100 * np.mean(dead):.0f}%' if dead else '-'):>9}")
    print(f"\n  frames between detected and estimated restarts: {total:,} "
          f"({100 * total / max(1, int((labels != 'X').sum())):.2f}% of graded)")
    print("  (negative = the real restart came BEFORE the estimate)")


if __name__ == "__main__":
    main()
