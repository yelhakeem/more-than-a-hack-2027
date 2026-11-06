"""v14: v13 without the interception rule (v12 + set-piece set-ups).

v13's interception rule backed a new team after 0.4 s of control. In practice many of those
"controls" were touches, blocks and duels, which CHALLENGE.md says are NOT changes of
possession: the full-match possession count jumped from 291 (v12) to 337 and wrong-team time
rose. v14 keeps v13's set-piece set-up rule and drops the interception rule.

Usage:
    python solution/v14.py   # writes evaluation/results/v14.csv
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402
import v12  # noqa: E402
import v13  # noqa: E402
from v1 import D  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v14.csv")


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
    setup_score, setup, slow, quick = v13.setup_clues(t, signals, speed, keeper_live)
    labels, fixed = v13.enforce_min_durations(v1.decode_halves(total + setup_score), setup)
    return labels, dict(setup=setup, slow=slow, quick=quick, **fixed)


def main() -> None:
    labels, stats = label_match(load_tracking())
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    graded = graded_mask(len(labels))
    print("\nCLUES USED")
    print(f"  set-ups, slow / quick:   {stats['slow']} / {stats['quick']}")
    print(f"  set-up frames:           {100 * stats['setup'][graded].mean():.1f}% of graded frames")
    print(f"  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v14.csv")
    print("then:  python solution/diagnose.py evaluation/results/v14.csv")


if __name__ == "__main__":
    main()
