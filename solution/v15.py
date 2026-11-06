"""v15: v12 plus team clues from Smart Tagging phase tags.

v2-v14 used only the 59 set-piece tags. The other 579 tags describe phases of play and each
names a team:
  - in possession:     Build Up, Progression, Creating Chances, Ball In Final Third,
                       Ball In The Box, Players In The Box, Long Ball, Attacking Transition
  - out of possession: Defending In Attacking / Middle / Defensive Third,
                       Defensive Transition (that team has just lost the ball)
Phases are not possessions and the tags are automatic, so each tag type is weighted by a
self-check without the answer key: on frames inside its intervals where possession is
visible (ball within 1 m of a player, not contested), how often does the implied team agree?
50% agreement gives weight 0, 75% or more the full weight.

Usage:
    python solution/v15.py   # writes evaluation/results/v15.csv
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
from tracking import HOME, graded_mask, load_tracking  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402
import v11  # noqa: E402
import v12  # noqa: E402
from v1 import A, H  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v15.csv")

IN_POSSESSION = ("build up", "progression", "creating chances", "ball in final third",
                 "ball in the box", "players in the box", "long ball", "attacking transition")
OUT_OF_POSSESSION = ("defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition")
PHASE_MAX_CLUE = 0.5
VISIBLE_RADIUS_M = 1.0


def phase_clues(t: dict, signals: dict, tags: list) -> tuple:
    """Team clue over every phase tag, weighted per tag type by visible-possession agreement."""
    n = len(t["bx"])
    graded = graded_mask(n)
    home_d, away_d = v11.nearest_by_team(t)
    contested = (home_d <= v1.CONTROL_RADIUS_M) & (away_d <= v1.CONTROL_RADIUS_M)
    visible = graded & (signals["dist"] <= VISIBLE_RADIUS_M) & ~signals["off"] & ~contested
    visible_side = np.where(signals["team"] == HOME, H, A)

    implied = {}                                        # code -> list of (start, end, state)
    for tag in tags:
        code = v2._norm(tag["code"])
        side = v2.team_side(tag["tags"].get("Team"))
        if side is None or code not in IN_POSSESSION + OUT_OF_POSSESSION:
            continue
        state = side if code in IN_POSSESSION else (A if side == H else H)
        implied.setdefault(code, []).append((tag["start_frame"], min(n, tag["end_frame"]), state))

    score = np.zeros((n, 3))
    report = {}
    for code, spans in sorted(implied.items()):
        agree = total = 0
        for a, b, state in spans:
            v = visible[a:b]
            agree += int((visible_side[a:b][v] == state).sum())
            total += int(v.sum())
        rate = agree / total if total else 0.5
        weight = float(np.clip((rate - 0.5) / 0.25, 0, 1)) * PHASE_MAX_CLUE
        for a, b, state in spans:
            score[a:b, state] += weight
        report[code] = (len(spans), total, rate, weight)
    return score, report


def label_match(t: dict) -> tuple:
    _, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    speed = v6.ball_speed(t)
    tags = load_tags()
    runners = v8.runners_present(t)

    total = v10.build_v9_total(t, signals)
    total += v10.crossing_clues(t, signals, speed)[0]
    total += v12.backfill_clues(t, signals, tags)[0]
    total += v12.adjustment_clues(t, signals, speed, runners)[0]
    phases, report = phase_clues(t, signals, tags)
    labels, fixed = v5.enforce_min_durations(v1.decode_halves(total + phases))
    return labels, dict(report=report, **fixed)


def main() -> None:
    labels, stats = label_match(load_tracking())
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print("\nPHASE TAGS, CHECKED ON FRAMES WHERE POSSESSION IS VISIBLE (no answer key)")
    print(f"  {'tag':<30} {'tags':>5} {'checked frames':>15} {'agrees':>7} {'weight':>7}")
    for code, (count, checked, rate, weight) in stats["report"].items():
        print(f"  {code:<30} {count:>5} {checked:>15,} {100 * rate:>6.1f}% {weight:>7.2f}")
    print(f"\n  blips removed:           {stats['short stoppages']} short stoppages, "
          f"{stats['short plays']} short plays")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v15.csv")
    print("then:  python solution/diagnose.py evaluation/results/v15.csv")


if __name__ == "__main__":
    main()
