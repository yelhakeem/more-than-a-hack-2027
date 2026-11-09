"""How well does the model (v19) detect changes of possession? Measured on the external DFL
matches only, each held out from training and degraded like our video.

A "change of possession" = the ball goes to the other team, either directly (H -> A) or across
a stoppage (H -> D -> A, e.g. a ball out restarted by the opponents). A detection counts if
the predicted change goes to the same team within 0.5 s or 1 s of the true one (the
competition's own tolerances, possession_metrics.match_transitions).

Nothing here reads our match or its answer key. It measures the model only: the rules in
v23 need Smart Tagging, which the DFL matches don't have.

Usage:
    python solution/dfl_report.py      # ~10 minutes
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl  # noqa: E402
import model18  # noqa: E402
import train_v19  # noqa: E402
from model18 import pm  # noqa: E402

TOLERANCES = (12, 25)          # frames: 0.5 s and 1 s


def team_changes(codes: np.ndarray, runs) -> list:
    """[(frame, from_team, to_team)] each time possession goes to the other team."""
    out = []
    for start, stop in runs:
        last = None
        for frame, _, to in pm.transitions(codes[start:stop], [(0, stop - start)]):
            if to in (pm.H, pm.A):
                if last is not None and to != last:
                    out.append((start + frame, last, to))
                last = to
        # the first team of a half is a kick-off, not a change
    return out


def joined(labels: np.ndarray, pred: np.ndarray, segments) -> tuple:
    """Halves joined with one X frame between them, as the scorer expects."""
    gt, pr = [], []
    for a, b in segments:
        gt += [labels[a:b], np.array([pm.X], dtype=np.int8)]
        pr += [pred[a:b], np.array([pm.X], dtype=np.int8)]
    return np.concatenate(gt[:-1]).astype(np.int8), np.concatenate(pr[:-1]).astype(np.int8)


def main() -> None:
    start = time.time()
    with open(train_v19.PARAMS_PATH) as handle:
        fees = json.load(handle)
    rng = np.random.default_rng(model18.SEED)
    totals = {tol: dict(matched=0, missed=0, spurious=0, lat=[]) for tol in TOLERANCES}
    counts = dict(true=0, pred=0, direct=0, restart=0)
    all_trans = {tol: [0, 0, 0] for tol in TOLERANCES}
    print("HELD-OUT DFL MATCHES (external data only)")
    for match_id in dfl.MATCHES:
        others = [m for m in dfl.MATCHES if m != match_id]
        X, y, _ = train_v19.training_rows(others, rng)
        model = model18.new_model().fit(X, y)
        score, labels, segments, _ = train_v19.held_out(match_id, model)
        pred = model18.decode(score, segments, fees["switch_fee"], fees["dead_fee"])
        gt, pr = joined(labels, pred, segments)
        runs = pm.valid_runs(gt)
        true_ch, pred_ch = team_changes(gt, runs), team_changes(pr, runs)
        counts["true"] += len(true_ch)
        counts["pred"] += len(pred_ch)
        for frame, _, _ in true_ch:
            counts["direct" if gt[frame - 1] in (pm.H, pm.A) else "restart"] += 1
        for tol in TOLERANCES:
            pairs, missed, spurious = pm.match_transitions(true_ch, pred_ch, tol, typed="to")
            totals[tol]["matched"] += len(pairs)
            totals[tol]["missed"] += len(missed)
            totals[tol]["spurious"] += len(spurious)
            totals[tol]["lat"] += [p[0] - g[0] for g, p in pairs]
            r = pm.transition_scores(pm.transitions(gt, runs), pm.transitions(pr, runs), tol)
            all_trans[tol][0] += r["n_matched"]
            all_trans[tol][1] += r["n_missed"]
            all_trans[tol][2] += r["n_spurious"]
        print(f"  {match_id}: {len(true_ch)} true changes, {len(pred_ch)} predicted "
              f"({time.time() - start:.0f} s)")

    print(f"\nCHANGES OF POSSESSION (7 held-out matches, {counts['true']:,} true changes: "
          f"{counts['direct']:,} direct, {counts['restart']:,} across a stoppage; "
          f"{counts['pred']:,} predicted)")
    print(f"  {'within':<8}{'found (recall)':>16}{'correct (precision)':>21}{'F1':>7}{'median delay':>14}")
    for tol in TOLERANCES:
        tt = totals[tol]
        rec = tt["matched"] / max(1, tt["matched"] + tt["missed"])
        prec = tt["matched"] / max(1, tt["matched"] + tt["spurious"])
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        delay = np.median(tt["lat"]) / 25 if tt["lat"] else float("nan")
        print(f"  {tol / 25:.1f} s   {100 * rec:>14.1f}%{100 * prec:>20.1f}%{f1:>7.2f}{delay:>+13.2f}s")
    print("\nALL STATE CHANGES (team changes plus every start and end of a stoppage)")
    for tol in TOLERANCES:
        m, mi, sp = all_trans[tol]
        rec, prec = m / max(1, m + mi), m / max(1, m + sp)
        print(f"  within {tol / 25:.1f} s: found {100 * rec:.1f}%, correct {100 * prec:.1f}%")


if __name__ == "__main__":
    main()
