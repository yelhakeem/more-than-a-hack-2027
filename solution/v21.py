"""v21: v20 with restarts found the way the Laws describe them, plus quick free kicks.

Changes from v20:
  1. Set-piece restart: v4 put every restart at 68% of its Smart Tagging window (measured on
     the two kick-offs only). v21 uses the restart restarts.py finds in the tracking: the ball
     back over the touchline (throw-in), or a still ball struck at a corner, in a goal area, on
     the centre spot, or anywhere (free kick). Falls back to 68% when none is found. Both the
     set-piece dead clue and the set-piece back-fill (v12) use it.
  2. Set-piece set-ups (v13/v14, Joe's rule): a still ball the opponents keep away from (the
     Laws require 9.15 m), or back away from (a quick free kick), is a placed ball: dead. v14
     dropped it only because it scored the same as v12 on the public window, which is no longer
     a reason we use.

No answer key, no video. The report shows each change on its own and both together.

Usage:
    python solution/v21.py   # writes evaluation/results/v21.csv
"""

import os
import sys
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from tracking import FPS, load_tracking  # noqa: E402
import restarts  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v4  # noqa: E402
import v5  # noqa: E402
import v6  # noqa: E402
import v7  # noqa: E402
import v8  # noqa: E402
import v10  # noqa: E402
import v12  # noqa: E402
import v13  # noqa: E402
import v17  # noqa: E402
import v20  # noqa: E402
from v1 import D  # noqa: E402

OUT_PATH = os.path.join(ROOT, "evaluation", "results", "v21.csv")


@contextmanager
def patched(pairs):
    """Temporarily replace module attributes; always restore them."""
    saved = [(m, name, getattr(m, name)) for m, name, _ in pairs]
    try:
        for m, name, value in pairs:
            setattr(m, name, value)
        yield
    finally:
        for m, name, value in saved:
            setattr(m, name, value)


def detected_restart_functions(found: dict):
    """Replacements for v7.restart_frames and v4.set_piece_clues that use detected restarts."""
    def restart_frames(tags):
        out = []
        for tag in tags:
            if v2._norm(tag["code"]) == "set pieces":
                frame = found.get(id(tag), (None,))[0]
                if frame is None:
                    start, end = tag["start_frame"], tag["end_frame"]
                    frame = start + int(round(v4.RESTART_FRACTION * (end - start)))
                out.append((frame, tag))
        return out

    def set_piece_clues(n_frames, tags):
        score = np.zeros((n_frames, 3))
        counts = {"set pieces": 0, "with team": 0}
        for restart, tag in restart_frames(tags):
            if v2._norm(tag["tags"].get("Type")) not in v2.SET_PIECE_TYPES:
                continue
            start, end = tag["start_frame"], tag["end_frame"]
            score[max(0, min(start, restart) - int(v2.BEFORE_TAG_S * FPS)):restart, D] += v2.SET_PIECE_DEAD_CLUE
            counts["set pieces"] += 1
            side = v2.team_side(tag["tags"].get("Team"))
            if side is not None:
                score[restart:min(n_frames, max(end, restart) + int(v2.AFTER_TAG_S * FPS)), side] += v2.RESTART_TEAM_CLUE
                counts["with team"] += 1
        return score, counts

    return restart_frames, set_piece_clues


def label_match(t: dict, use_restarts: bool = True, use_setups: bool = True,
                finder=restarts.find_restarts) -> tuple:
    _, signals = v1.clues(t)
    _, signals["team"] = v1.nearest_player(t)
    speed = v6.ball_speed(t)
    tags = load_tags()
    runners = v8.runners_present(t)
    found = finder(t, speed, signals["off"], tags, v2._norm)
    rf, spc = detected_restart_functions(found)
    pairs = [(m, name, value) for m, name, value in v17.TIMINGS]
    if use_restarts:
        pairs += [(v7, "restart_frames", rf), (v4, "set_piece_clues", spc)]
    with patched(pairs):
        total = v10.build_v9_total(t, signals)
        total += v10.crossing_clues(t, signals, speed)[0]
        total += v12.backfill_clues(t, signals, tags)[0]
        total += v12.adjustment_clues(t, signals, speed, runners)[0]
        setup = np.zeros(len(speed), dtype=bool)
        if use_setups:
            keeper_live = v7.keeper_clues(t, signals, speed)[0][:, D] < 0
            setup_score, setup, _, _ = v13.setup_clues(t, signals, speed, keeper_live)
            total += setup_score
        labels, _ = v13.enforce_min_durations(v1.decode_halves(total), setup)
    labels, _, _ = v20.remove_switch_blips(labels, signals["off"])
    stats = dict(detected=sum(1 for f in found.values() if f[1]), tags=len(found),
                 setup_pct=100 * setup.mean(), found=found, tag_list=tags)
    return labels, stats


def main() -> None:
    t = load_tracking()
    variants = [("v20", dict(use_restarts=False, use_setups=False)),
                ("+restarts", dict(use_restarts=True, use_setups=False)),
                ("+set-ups", dict(use_restarts=False, use_setups=True)),
                ("v21", dict(use_restarts=True, use_setups=True))]
    results = {}
    for name, kw in variants:
        labels, stats = label_match(t, **kw)
        results[name] = (labels, stats, v17.label_free_checks(t, labels), v20.stoppage_lengths(labels))
    labels, stats, _, _ = results["v21"]
    v1.write_submission(OUT_PATH, labels)
    print(f"wrote {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"\nRESTARTS DETECTED: {stats['detected']} of {stats['tags']} set-piece tags; "
          f"set-up frames {stats['setup_pct']:.1f}% of all frames")

    lo, hi = v17.PUBLISHED_STOPPED_PCT
    print("\nLABEL-FREE CHECKS (no answer key)" + "".join(f"{n:>11}" for n, _ in variants))
    rows = [("possessions", lambda c, s: f"{c['possessions']}"),
            ("stoppages (published 109-117)", lambda c, s: f"{c['stoppages']}"),
            (f"dead % (published {lo}-{hi})", lambda c, s: f"{c['dead_pct']:.1f}%"),
            ("agrees with visible possession", lambda c, s: f"{c['visible_agree']:.1f}%"),
            ("ball off pitch labelled dead", lambda c, s: f"{c['off_dead']:.1f}%"),
            ("stoppages under 3 s", lambda c, s: f"{int((s < 3).sum())}"),
            ("median stoppage (pub. mean 23.1 s)", lambda c, s: f"{np.median(s):.1f}s")]
    for label, fn in rows:
        print(f"  {label:<34}" + "".join(f"{fn(results[n][2], results[n][3]):>11}" for n, _ in variants))
    v1.report(labels)
    print("\nNext:  python evaluation/validate_submission.py evaluation/results/v21.csv")
    print("then:  python solution/compare.py v20 v21")


if __name__ == "__main__":
    main()
