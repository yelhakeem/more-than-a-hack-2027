"""v22: v21 with a sturdier restart finder (restarts22.py).

Same pipeline as v21 (v20 + detected restarts + set-ups). Only the restart finder changes:
a jitter-proof still-ball test, corner balls allowed just over the line, and a fallback
measured from this match's own detected restarts instead of 68% of a 35 s window.

Check without the answer key: the two "Kick Off (Start)" tags must be detected at the
published kick-off frames (DATA.md / FAQ.md: 11,185 and 113,742).

Usage:
    python solution/v22.py   # writes evaluation/results/v22.csv
"""

import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from tracking import FPS, load_tracking  # noqa: E402
import restarts22  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v7  # noqa: E402
import v17  # noqa: E402
import v20  # noqa: E402
import v21  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v22.csv")


def main() -> None:
    t = load_tracking()
    old, _ = v21.label_match(t)
    labels, stats = v21.label_match(t, finder=restarts22.find_restarts)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")

    found, tags = stats["found"], stats["tag_list"]
    per_type, hit = Counter(), Counter()
    print("\nKICK-OFF CHECK (published frames, not the answer key)")
    for tag in tags:
        if id(tag) not in found:
            continue
        kind = v2._norm(tag["tags"].get("Type"))
        frame, detected = found[id(tag)]
        per_type[kind] += 1
        hit[kind] += detected
        if kind == "kick off (start)":
            target = min(v7.KICKOFFS.values(), key=lambda k: abs(k - tag["start_frame"]))
            status = f"detected at {frame:,}" if detected else f"NOT detected (fallback {frame:,})"
            print(f"  kick-off near {target:,}: {status}, off by {(frame - target) / FPS:+.2f} s")
    print("\nRESTARTS DETECTED BY TYPE")
    for kind in sorted(per_type):
        print(f"  {kind:<24}{hit[kind]:>3} of {per_type[kind]}")

    a, b = v17.label_free_checks(t, old), v17.label_free_checks(t, labels)
    sa, sb = v20.stoppage_lengths(old), v20.stoppage_lengths(labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    print("\nLABEL-FREE CHECKS (no answer key)               v21       v22")
    print(f"  possessions                               {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages (published 109-117)             {a['stoppages']:>7}   {b['stoppages']:>7}")
    print(f"  dead % (published {lo}-{hi})              {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch labelled dead              {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    print(f"  stoppages under 3 s                       {int((sa < 3).sum()):>7}   {int((sb < 3).sum()):>7}")
    print(f"  median stoppage (pub. mean 23.1 s)        {np.median(sa):>6.1f}s   {np.median(sb):>6.1f}s")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v22.csv")
    print("then:  python solution/compare.py v21 v22")


if __name__ == "__main__":
    main()
