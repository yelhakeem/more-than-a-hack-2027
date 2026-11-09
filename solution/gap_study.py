"""v23's one tuned number, chosen on the external DFL matches only: how long must a ball gap be
before the model (v19) reads it better than the rules do?

Inside a ball gap the rules mostly carry on with the state they had when the ball vanished.
So on each held-out DFL match (degraded like our video, model trained on the other six) this
compares, gap by gap:
  - the model: v19's decoded labels inside the gap;
  - "carry on": the TRUE label at the moment the ball vanished, held for the whole gap. This
    is generous to the rules (they don't always know the true state at that moment).
The shortest gap length from which the model wins in every longer length band too becomes
MIN_GAP_S, saved to solution/models/v23_params.json.

Nothing here reads our match or its answer key.

Usage:
    python solution/gap_study.py      # ~10 minutes (retrains the 7 held-out models)
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

PARAMS_PATH = os.path.join(train_v19.MODEL_DIR, "v23_params.json")
BANDS_S = (0, 2, 5, 10, 20, 40, 80, 1e9)
FPS = 25


def gap_runs(missing: np.ndarray, segments) -> list:
    """(start, stop) of ball gaps, never crossing a half."""
    out = []
    for a, b in segments:
        m = missing[a:b].astype(np.int8)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], m, [0]))))
        out += [(a + s, a + e) for s, e in zip(edges[::2], edges[1::2])]
    return out


def main() -> None:
    start = time.time()
    with open(train_v19.PARAMS_PATH) as handle:
        fees = json.load(handle)
    rng = np.random.default_rng(model18.SEED)
    bands = {i: [0, 0, 0] for i in range(len(BANDS_S) - 1)}      # frames, model right, carry right
    print("HELD-OUT DFL MATCHES (external data only)")
    for match_id in dfl.MATCHES:
        others = [m for m in dfl.MATCHES if m != match_id]
        X, y, _ = train_v19.training_rows(others, rng)
        model = model18.new_model().fit(X, y)
        score, labels, segments, _ = train_v19.held_out(match_id, model)
        degraded = dfl.degrade(dfl.load_match(match_id),
                               np.random.default_rng(1000 + dfl.MATCHES.index(match_id)),
                               train_v19.TEST_MISSING)
        pred = model18.decode(score, segments, fees["switch_fee"], fees["dead_fee"])
        for a, b in gap_runs(np.isnan(degraded["bx"]), segments):
            if a == segments[0][0] or any(a == s for s, _ in segments):
                continue                                         # no state before the gap
            length = (b - a) / FPS
            band = int(np.searchsorted(BANDS_S, length, side="right") - 1)
            truth = labels[a:b]
            bands[band][0] += b - a
            bands[band][1] += int(np.sum(pred[a:b] == truth))
            bands[band][2] += int(np.sum(truth == labels[a - 1]))
        print(f"  {match_id}: done ({time.time() - start:.0f} s)")

    print("\nINSIDE BALL GAPS: share of frames labelled right")
    print(f"  {'gap length':<12}{'frames':>10}{'model':>9}{'carry on':>10}")
    wins = []
    for i in range(len(BANDS_S) - 1):
        frames, model_ok, carry_ok = bands[i]
        lo, hi = BANDS_S[i], BANDS_S[i + 1]
        label = f"{lo:g}-{hi:g}s" if hi < 1e8 else f"{lo:g}s+"
        if frames == 0:
            print(f"  {label:<12}{0:>10}")
            continue
        m, c = model_ok / frames, carry_ok / frames
        wins.append((lo, m > c))
        print(f"  {label:<12}{frames:>10,}{100 * m:>8.1f}%{100 * c:>9.1f}%")
    min_gap = None
    for k, (lo, _) in enumerate(wins):
        if all(w for _, w in wins[k:]):
            min_gap = lo
            break
    if min_gap is None:
        print("\n  The model never beats carrying on, in every longer band: v23 would change nothing.")
        min_gap = float("inf")
    else:
        print(f"\n  Model wins from {min_gap:g} s gaps upwards -> MIN_GAP_S = {min_gap:g}")
    with open(PARAMS_PATH, "w") as handle:
        json.dump({"min_gap_s": float(min_gap) if np.isfinite(min_gap) else None,
                   "bands_s": [float(v) for v in BANDS_S[:-1]],
                   "frames_model_carry": [[int(v) for v in bands[i]] for i in range(len(BANDS_S) - 1)]},
                  handle, indent=2)
    print(f"  saved {os.path.relpath(PARAMS_PATH, os.path.dirname(HERE))}  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
