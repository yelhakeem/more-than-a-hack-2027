"""Train v19's model and choose its two decoder fees, using ONLY the external DFL matches.

  1. Each DFL match is degraded to look like our tactical-video tracking (dfl.degrade): ball
     gaps with the same length distribution as ours, placed so the ball is seen off the pitch
     as rarely as in our video, plus player gaps. It is also used from the other team's side
     (dfl.swap_teams), so the model doesn't learn "home" or "away" habits of particular clubs.
     Changes from v18: realistic gap lengths and placement (v18 used short random gaps, so it
     never learned that a ball which leaves the pitch usually vanishes), time-since-seen
     features capped at 60 s instead of 30 s, and higher stoppage fees in the search.
  2. Leave one match out, seven times: train on six matches, predict the seventh.
  3. Pick the switch / dead fees that give the best mean spellF1 over the seven held-out
     matches (the competition's own metric, computed on DFL labels).
  4. Train the final model on all seven and save it with the chosen fees.

Nothing in this script reads our match or its answer key.

Usage:
    python solution/train_v19.py      # ~15-25 minutes; writes solution/models/v19.joblib
"""

import json
import os
import sys
import time

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl  # noqa: E402
import features18  # noqa: E402
import model18  # noqa: E402

MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "v19.joblib")
PARAMS_PATH = os.path.join(MODEL_DIR, "v19_params.json")
SUBSAMPLE = 5                               # neighbouring frames are near-copies
SWITCH_FEES = (2, 4, 6, 8, 12, 16, 24, 32)
DEAD_FEES = (4, 8, 12, 16, 24, 32, 48)
CAP_S = 60.0                               # longest ball gap in our tracking is 221 s
TEST_MISSING = dfl.BALL_MISSING_SHARE      # held-out matches: our match's ball coverage


def training_rows(match_ids, rng) -> tuple:
    Xs, ys = [], []
    for match_id in match_ids:
        clean = dfl.load_match(match_id)
        for view in (clean, dfl.swap_teams(clean)):
            t = dfl.degrade(view, rng)
            X, names = features18.build(t, [tuple(s) for s in t["segments"]], CAP_S)
            Xs.append(X[::SUBSAMPLE])
            ys.append(t["label"][::SUBSAMPLE])
    return np.concatenate(Xs), np.concatenate(ys), names


def held_out(match_id: str, model) -> tuple:
    t = dfl.degrade(dfl.load_match(match_id), np.random.default_rng(1000 + dfl.MATCHES.index(match_id)),
                    TEST_MISSING)
    segments = [tuple(s) for s in t["segments"]]
    X, _ = features18.build(t, segments, CAP_S)
    return model18.log_probs(model, X), t["label"], segments, t["degrade_info"]


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(model18.SEED)
    folds = []
    print("LEAVE ONE MATCH OUT (external DFL data only)")
    for match_id in dfl.MATCHES:
        others = [m for m in dfl.MATCHES if m != match_id]
        X, y, names = training_rows(others, rng)
        model = model18.new_model().fit(X, y)
        score, labels, segments, info = held_out(match_id, model)
        frame_acc = float(np.mean(np.argmax(score, axis=1) == labels))
        folds.append((match_id, score, labels, segments))
        print(f"  {match_id}: trained on {len(y):,} frames, held-out per-frame accuracy "
              f"{100 * frame_acc:.1f}%  (ball missing {100 * info[3]:.1f}%, seen off pitch "
              f"{100 * info[2]:.1f}%, gaps at exits {100 * info[1]:.0f}%; {time.time() - start:.0f} s)")

    print("\nCHOOSING THE FEES (mean spellF1 over the 7 held-out matches)")
    results = []
    for sw in SWITCH_FEES:
        for de in DEAD_FEES:
            per = [model18.score_external(lab, model18.decode(sc, seg, sw, de), seg)
                   for _, sc, lab, seg in folds]
            mean = {k: float(np.mean([p[k] for p in per])) for k in per[0]}
            results.append((mean["spell_f1_mean"], sw, de, mean, per))
    results.sort(key=lambda r: -r[0])
    print(f"  {'switch':>6} {'dead':>5} {'spellF1':>8} {'macroF1':>8} {'acc':>6} {'spells':>7} {'true':>6}")
    for f1, sw, de, mean, _ in results[:8]:
        print(f"  {sw:>6} {de:>5} {f1:>8.3f} {mean['macro_f1']:>8.3f} {100 * mean['accuracy']:>5.1f}% "
              f"{mean['n_pred_spells']:>7.0f} {mean['n_gt_spells']:>6.0f}")
    best_f1, best_sw, best_de, _, per = results[0]
    print("\n  per held-out match at the chosen fees:")
    for (match_id, *_), p in zip(folds, per):
        print(f"    {match_id}: spellF1 {p['spell_f1_mean']:.3f}  macroF1 {p['macro_f1']:.3f}  "
              f"spells {p['n_pred_spells']} vs {p['n_gt_spells']}")

    print("\nFINAL MODEL (all 7 matches)")
    X, y, names = training_rows(dfl.MATCHES, rng)
    model = model18.new_model().fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    params = {"switch_fee": best_sw, "dead_fee": best_de, "features": names, "cap_s": CAP_S,
              "heldout_spell_f1": best_f1, "trained_frames": int(len(y))}
    with open(PARAMS_PATH, "w") as handle:
        json.dump(params, handle, indent=2)
    print(f"  trained on {len(y):,} frames, {len(names)} features")
    print(f"  fees: switch {best_sw}, dead {best_de}")
    print(f"  saved {os.path.relpath(MODEL_PATH, os.path.dirname(HERE))} and "
          f"{os.path.relpath(PARAMS_PATH, os.path.dirname(HERE))}  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
