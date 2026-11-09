"""v23: v22, with the v19 model deciding inside long ball gaps.

About 40% of our frames have no ball, mostly in long stretches (median gap 10.7 s). There the
rules have little to go on and mostly carry on with the last state. The model (v19, trained
only on external Bundesliga matches) reads the players instead. gap_study.py measured, on the
external matches only, from what gap length the model beats "carry on": MIN_GAP_S.

Change: inside every ball gap at least MIN_GAP_S long, v22's labels are replaced by v19's,
except inside Smart Tagging set-piece windows and goal windows (direct evidence the model
doesn't see). Then the usual cleanup (v5 minimum durations, v20 switch blips).

No answer key, no video.

Usage:
    python solution/gap_study.py   # once: writes solution/models/v23_params.json
    python solution/v23.py         # writes evaluation/results/v23.csv
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, half_video_blocks  # noqa: E402
from tracking import FPS, load_tracking  # noqa: E402
import gap_study  # noqa: E402
import restarts22  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v5  # noqa: E402
import v17  # noqa: E402
import v19  # noqa: E402
import v20  # noqa: E402
import v21  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v23.csv")
GOAL_WINDOW_S = 41.0                 # v17's value


def protected(n: int, tags: list) -> np.ndarray:
    """Frames with direct tag evidence: set-piece windows and the run-up to goal kick-offs."""
    mask = np.zeros(n, dtype=bool)
    for tag in tags:
        kind = v2._norm(tag["tags"].get("Type"))
        if v2._norm(tag["code"]) == "set pieces" and kind in v2.SET_PIECE_TYPES:
            mask[max(0, tag["start_frame"] - int(v2.BEFORE_TAG_S * FPS)):
                 min(n, tag["end_frame"] + int(v2.AFTER_TAG_S * FPS))] = True
        if kind == "kick off (after goal)":
            mask[max(0, tag["start_frame"] - int(GOAL_WINDOW_S * FPS)):tag["start_frame"]] = True
    return mask


def label_match(t: dict, min_gap_s: float) -> tuple:
    rules, stats = v21.label_match(t, finder=restarts22.find_restarts)
    if min_gap_s is None:
        return rules, rules, dict(gaps=0, frames=0)
    model, _ = v19.label_match(t)
    n = len(rules)
    keep = protected(n, stats["tag_list"])
    blocks = [(first, last + 1) for _, first, last in half_video_blocks(GAME)]
    labels = rules.copy()
    gaps = frames = 0
    for a, b in gap_study.gap_runs(np.isnan(t["bx"]), blocks):
        if (b - a) < min_gap_s * FPS:
            continue
        take = np.arange(a, b)[~keep[a:b] & (model[a:b] != "X")]
        labels[take] = model[take]
        gaps += 1
        frames += int(np.sum(rules[take] != model[take]))
    labels, _ = v5.enforce_min_durations(labels)
    _, signals = v1.clues(t)
    labels, _, _ = v20.remove_switch_blips(labels, signals["off"])
    return rules, labels, dict(gaps=gaps, frames=frames)


def main() -> None:
    with open(gap_study.PARAMS_PATH) as handle:
        min_gap = json.load(handle)["min_gap_s"]
    t = load_tracking()
    rules, labels, stats = label_match(t, min_gap)
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    if min_gap is None:
        print("\nMIN_GAP_S: none (the model never beat carrying on in the external study) -> v23 = v22")
    else:
        print(f"\nMIN_GAP_S = {min_gap:g} s (from gap_study.py, external data): "
              f"{stats['gaps']} ball gaps handed to the model, {stats['frames']:,} frames changed")

    a, b = v17.label_free_checks(t, rules), v17.label_free_checks(t, labels)
    sa, sb = v20.stoppage_lengths(rules), v20.stoppage_lengths(labels)
    lo, hi = v17.PUBLISHED_STOPPED_PCT
    print("\nLABEL-FREE CHECKS (no answer key)               v22       v23")
    print(f"  possessions                               {a['possessions']:>7}   {b['possessions']:>7}")
    print(f"  stoppages (published 109-117)             {a['stoppages']:>7}   {b['stoppages']:>7}")
    print(f"  dead % (published {lo}-{hi})              {a['dead_pct']:>6.1f}%   {b['dead_pct']:>6.1f}%")
    print(f"  agrees with visible possession            {a['visible_agree']:>6.1f}%   {b['visible_agree']:>6.1f}%")
    print(f"  ball off pitch labelled dead              {a['off_dead']:>6.1f}%   {b['off_dead']:>6.1f}%")
    print(f"  stoppages under 3 s                       {int((sa < 3).sum()):>7}   {int((sb < 3).sum()):>7}")
    print(f"  median stoppage (pub. mean 23.1 s)        {np.median(sa):>6.1f}s   {np.median(sb):>6.1f}s")
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v23.csv")
    print("then:  python solution/compare.py v22 v23")


if __name__ == "__main__":
    main()
