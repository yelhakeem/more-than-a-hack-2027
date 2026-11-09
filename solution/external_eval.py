"""Score the full pipelines (v17, v20, v22, v23, v24) and the model alone (v19) on the 7 external DFL
matches, with the competition's own metrics.

For each DFL match, held out from the model's training:
  1. Degrade the tracking like our video (dfl.degrade, the same seed as train_v19).
  2. Build stand-in Smart Tagging tags from the DFL event data: one 35 s "Set Pieces" window per
     throw-in, corner, goal kick and free kick, and a 5 s window per kick-off, placed like ours
     (setpiece_check: restarts sit ~7.6 s into the 35 s windows; 68% into the kick-off ones).
     Throw-ins and the opening kick-offs carry no team, as in our Smart Tagging; the others
     name the team taking them.
  3. Run each version's code unchanged, pointed at this match instead of ours (dfl_context).
  4. Score against the DFL labels: spellF1, macroF1, per-frame accuracy, and how many changes
     of possession are found within 0.5 s and 1 s.

External data only: no file of our match, its answer key or its video is opened.

Caveats (for the technical document): DFL tracking is degraded synthetically; the stand-in tags
are built from event data, not by Smart Tagging; v23's MIN_GAP_S was chosen on these same 7
matches (one number), and DFL's possession definition is finer than the challenge's.

Usage:
    python solution/external_eval.py      # ~1 hour
"""

import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import dfl  # noqa: E402
import dfl_report  # noqa: E402
import features18  # noqa: E402
import gap_study  # noqa: E402
import model18  # noqa: E402
import restarts22  # noqa: E402
import train_v19  # noqa: E402
import tracking  # noqa: E402
import v1  # noqa: E402
import v7  # noqa: E402
import v17  # noqa: E402
import v19  # noqa: E402
import v20  # noqa: E402
import v21  # noqa: E402
import v23  # noqa: E402
import v24  # noqa: E402
import v25  # noqa: E402
from model18 import pm  # noqa: E402

FPS = 25
SET_WINDOW_S, SET_LEAD_S = 35.0, 7.6
KICKOFF_WINDOW_S = 5.0
TEAM_NAMES = {dfl.H: "Barcelona", dfl.A: "Athletic"}     # what v2.team_side understands
VERSIONS = ("v17", "v20", "v22", "v23", "v24", "v25", "v19")


def standin_tags(match_id: str, segments) -> list:
    tags = []
    starts = {int(a) for a, _ in segments}
    for frame, kind, team in dfl.load_events(match_id):
        if kind == "Kick Off":
            kind = "Kick Off (Start)" if any(abs(frame - s) < 5 * FPS for s in starts) else "Kick Off (After Goal)"
            start = frame - int(round(0.68 * KICKOFF_WINDOW_S * FPS))
            end = start + int(KICKOFF_WINDOW_S * FPS)
        else:
            start = frame - int(SET_LEAD_S * FPS)
            end = start + int(SET_WINDOW_S * FPS)
        side = "N/A" if kind in ("Throw-in", "Kick Off (Start)") else TEAM_NAMES[team]
        tags.append({"code": "Set Pieces", "start_frame": max(0, start), "end_frame": end,
                     "start_s": start / FPS, "end_s": end / FPS,
                     "tags": {"Type": kind, "Team": side}})
    return tags


def prepare(t: dict) -> dict:
    """Add the columns our tracking has: player speeds and the (mostly empty) ball-speed column."""
    out = dict(t)
    out["ps"] = features18._player_speed(t["px"].astype(np.float64), t["py"].astype(np.float64))
    out["bs"] = np.full(len(t["bx"]), np.nan, dtype=np.float32)
    return out


@contextmanager
def dfl_context(segments, tags, model, fees):
    """Point every version's code at a DFL match: its halves, kick-offs, tags and model."""
    blocks = [(i + 1, int(a), int(b) - 1) for i, (a, b) in enumerate(segments)]
    n = int(segments[-1][1])

    def half_video_blocks(_game=None):
        return blocks

    def graded_mask(n_frames):
        mask = np.zeros(n_frames, dtype=bool)
        for _, first, last in blocks:
            mask[first:last + 1] = True
        return mask

    def v19_labels(t):
        segs = [(int(a), int(b)) for a, b in segments]
        X, _ = features18.build(t, segs, train_v19.CAP_S)
        path = model18.decode(model18.log_probs(model, X), segs, fees["switch_fee"], fees["dead_fee"])
        labels = np.full(n, "X", dtype="<U1")
        labels[path >= 0] = v1.STATES[path[path >= 0]]
        return labels, fees

    swaps = [(v1, "SECOND_HALF_KICKOFF", int(segments[1][0])),
             (v7, "KICKOFFS", {1: int(segments[0][0]), 2: int(segments[1][0])}),
             (v19, "label_match", v19_labels)]
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if name == "tracking" or name.startswith("v") or name.startswith("restarts"):
            for attr, value in (("half_video_blocks", half_video_blocks), ("graded_mask", graded_mask),
                                ("load_tags", lambda *a, **k: tags)):
                if hasattr(module, attr):
                    swaps.append((module, attr, value))
    saved = [(m, a, getattr(m, a)) for m, a, _ in swaps]
    try:
        for m, a, value in swaps:
            setattr(m, a, value)
        yield
    finally:
        for m, a, value in saved:
            setattr(m, a, value)


def to_codes(labels: np.ndarray) -> np.ndarray:
    return pm.label_codes(labels)


def score(truth: np.ndarray, labels: np.ndarray, segments) -> dict:
    pred = to_codes(labels)
    gt, pr = dfl_report.joined(truth, pred, segments)
    r = pm.evaluate(gt, pr)
    runs = pm.valid_runs(gt)
    out = {k: r[k] for k in ("spell_f1_mean", "macro_f1", "accuracy", "n_gt_spells", "n_pred_spells")}
    tc, pc = dfl_report.team_changes(gt, runs), dfl_report.team_changes(pr, runs)
    for tol in dfl_report.TOLERANCES:
        pairs, missed, spurious = pm.match_transitions(tc, pc, tol, typed="to")
        out[f"found_{tol}"] = len(pairs) / max(1, len(pairs) + len(missed))
        out[f"correct_{tol}"] = len(pairs) / max(1, len(pairs) + len(spurious))
    return out


def main() -> None:
    start = time.time()
    with open(train_v19.PARAMS_PATH) as handle:
        fees = json.load(handle)
    with open(gap_study.PARAMS_PATH) as handle:
        min_gap = json.load(handle)["min_gap_s"]
    rng = np.random.default_rng(model18.SEED)
    results = {v: [] for v in VERSIONS}
    print("EXTERNAL EVALUATION (7 DFL matches, each held out; external data only)")
    for match_id in dfl.MATCHES:
        others = [m for m in dfl.MATCHES if m != match_id]
        X, y, _ = train_v19.training_rows(others, rng)
        model = model18.new_model().fit(X, y)
        t = prepare(dfl.degrade(dfl.load_match(match_id),
                                np.random.default_rng(1000 + dfl.MATCHES.index(match_id)),
                                train_v19.TEST_MISSING))
        segments = [tuple(int(v) for v in s) for s in t["segments"]]
        tags = standin_tags(match_id, segments)
        with dfl_context(segments, tags, model, fees):
            base, _ = v17.label_match(t)
            _, signals = v1.clues(t)
            labels = {"v17": base,
                      "v20": v20.remove_switch_blips(base, signals["off"])[0],
                      "v22": v21.label_match(t, finder=restarts22.find_restarts)[0],
                      "v23": v23.label_match(t, min_gap)[1],
                      "v24": v24.label_match(t)[0],
                      "v25": v25.label_match(t)[0],
                      "v19": v19.label_match(t)[0]}
        for v in VERSIONS:
            results[v].append(score(t["label"], labels[v], segments))
        r = results["v24"][-1]
        print(f"  {match_id}: {len(tags)} stand-in tags; v24 spellF1 {r['spell_f1_mean']:.3f}, "
              f"macroF1 {r['macro_f1']:.3f}, accuracy {100 * r['accuracy']:.1f}%  "
              f"({time.time() - start:.0f} s)")

    print("\nMEAN OVER THE 7 HELD-OUT MATCHES")
    print(f"  {'version':<8}{'spellF1':>9}{'macroF1':>9}{'accuracy':>10}{'possessions':>13}{'true':>6}"
          f"{'changes found 0.5s / 1s':>26}{'correct 0.5s / 1s':>20}")
    for v in VERSIONS:
        rs = results[v]
        m = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
        print(f"  {v:<8}{m['spell_f1_mean']:>9.3f}{m['macro_f1']:>9.3f}{100 * m['accuracy']:>9.1f}%"
              f"{m['n_pred_spells']:>13.0f}{m['n_gt_spells']:>6.0f}"
              f"{100 * m['found_12']:>15.1f}% / {100 * m['found_25']:.1f}%"
              f"{100 * m['correct_12']:>11.1f}% / {100 * m['correct_25']:.1f}%")
    print(f"\n  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
