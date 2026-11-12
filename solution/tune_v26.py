"""Build v26 entirely on the 7 external DFL matches: train, measure, tune, and test fairly.

  1. Leave one match out: for each DFL match, train two models on the other six (v19's features,
     and v26's features with stand-in Smart Tagging) and predict the held-out match, degraded
     like our video.
  2. Ablation: v19's model alone, then each of v25's hard overrides added on its own.
  3. Tuning: coordinate search over the decoder fees, the set-piece and goal weights (soft or
     hard), how long before a restart the ball is dead, and the cleanup, maximising mean
     spellF1. Law 9 is not tuned: a ball over the line is out by definition, so it stays hard.
  4. Fair external score: the tuning is repeated with each match left out of it, and scored on
     that match only, so no match is scored with settings chosen on it.
  5. Final model trained on all 7; model and settings saved to solution/models/.

External data only: nothing here opens our match, its answer key or its video.

Usage:
    python solution/tune_v26.py      # ~1-1.5 hours
"""

import json
import os
import sys
import time

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import dfl  # noqa: E402
import dfl_report  # noqa: E402
import external_eval  # noqa: E402
import features18  # noqa: E402
import features26  # noqa: E402
import model18  # noqa: E402
import pipeline26 as P  # noqa: E402
import restarts22  # noqa: E402
import train_v19  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v6  # noqa: E402
from model18 import pm  # noqa: E402

MODEL_PATH = os.path.join(train_v19.MODEL_DIR, "v26.joblib")
PARAMS_PATH = os.path.join(train_v19.MODEL_DIR, "v26_params.json")
SUBSAMPLE = train_v19.SUBSAMPLE
CAP_S = train_v19.CAP_S
GRID = {
    "switch_fee": (4, 6, 8, 12, 16),
    "dead_fee": (8, 16, 24, 32, 48),
    "w_set": (0.0, 0.5, 1.0, 2.0, P.HARD),
    "set_lead_s": (2.0, 4.0, 8.0, 12.0),
    "w_goal": (0.0, P.HARD),
    "cleanup": (0, 1),
}
PASSES = 2
SWAP_NAMES = {"Barcelona": "Athletic", "Athletic": "Barcelona"}


def match_inputs(t: dict, match_id: str, swapped: bool) -> tuple:
    """Prepared tracking, segments, stand-in tags, restarts and off-pitch mask for one view."""
    t = external_eval.prepare(t)
    segments = [tuple(int(v) for v in s) for s in t["segments"]]
    tags = external_eval.standin_tags(match_id, segments)
    if swapped:
        for tag in tags:
            tag["tags"]["Team"] = SWAP_NAMES.get(tag["tags"]["Team"], tag["tags"]["Team"])
    _, signals = v1.clues(t)
    found = restarts22.find_restarts(t, v6.ball_speed(t), signals["off"], tags, v2._norm)
    return t, segments, tags, found, signals["off"]


def rows(match_ids, rng) -> tuple:
    """Training rows for both feature sets from degraded views of the given matches."""
    X19, X26, ys = [], [], []
    for match_id in match_ids:
        clean = dfl.load_match(match_id)
        for swapped, view in ((False, clean), (True, dfl.swap_teams(clean))):
            t, segments, tags, found, off = match_inputs(dfl.degrade(view, rng), match_id, swapped)
            a, names19 = features18.build(t, segments, CAP_S)
            b, names26 = features26.build(t, segments, tags, found, off, CAP_S)
            X19.append(a[::SUBSAMPLE])
            X26.append(b[::SUBSAMPLE])
            ys.append(t["label"][::SUBSAMPLE])
    return np.concatenate(X19), np.concatenate(X26), np.concatenate(ys), names26


def held_out(match_id: str) -> tuple:
    t = dfl.degrade(dfl.load_match(match_id), np.random.default_rng(1000 + dfl.MATCHES.index(match_id)),
                    train_v19.TEST_MISSING)
    t, segments, tags, found, off = match_inputs(t, match_id, False)
    x19, _ = features18.build(t, segments, CAP_S)
    x26, _ = features26.build(t, segments, tags, found, off, CAP_S)
    m = P.masks(t, tags, found, off, v2._norm, v2.SET_PIECE_TYPES)
    return t, segments, x19, x26, m


def score(truth: np.ndarray, codes: np.ndarray, segments) -> dict:
    gt, pr = dfl_report.joined(truth, codes, segments)
    r = pm.evaluate(gt, pr)
    runs = pm.valid_runs(gt)
    pairs, missed, spurious = pm.match_transitions(dfl_report.team_changes(gt, runs),
                                                   dfl_report.team_changes(pr, runs), 25, typed="to")
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "acc": r["accuracy"],
            "poss": r["n_pred_spells"], "true": r["n_gt_spells"],
            "found": len(pairs) / max(1, len(pairs) + len(missed)),
            "correct": len(pairs) / max(1, len(pairs) + len(spurious))}


def evaluate(folds: list, which: str, params: dict) -> list:
    return [score(f["truth"], P.label(f[which], f["masks"], f["segments"], params), f["segments"])
            for f in folds]


def mean(results: list, key: str = "spell") -> float:
    return float(np.mean([r[key] for r in results]))


def tune(folds: list, which: str, start: dict, log=print) -> dict:
    """Coordinate search: one setting at a time, keep whatever raises mean spellF1."""
    best = dict(start)
    best_score = mean(evaluate(folds, which, best))
    for _ in range(PASSES):
        changed = False
        for key, values in GRID.items():
            for value in values:
                if value == best[key]:
                    continue
                trial = dict(best, **{key: value})
                s = mean(evaluate(folds, which, trial))
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            log(f"    {key:<11} -> {best[key]!s:<7} (spellF1 {best_score:.3f})")
        if not changed:
            break
    return best


def line(name: str, results: list) -> str:
    return (f"  {name:<34}{mean(results):>8.3f}{mean(results, 'macro'):>9.3f}{100 * mean(results, 'acc'):>9.1f}%"
            f"{mean(results, 'poss'):>7.0f}{100 * mean(results, 'found'):>9.1f}%{100 * mean(results, 'correct'):>9.1f}%")


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(model18.SEED)
    folds = []
    print("1. LEAVE ONE MATCH OUT (external DFL data only)")
    for match_id in dfl.MATCHES:
        others = [m for m in dfl.MATCHES if m != match_id]
        X19, X26, y, names = rows(others, rng)
        m19 = model18.new_model().fit(X19, y)
        m26 = model18.new_model().fit(X26, y)
        t, segments, x19, x26, masks = held_out(match_id)
        folds.append({"id": match_id, "truth": t["label"], "segments": segments, "masks": masks,
                      "p19": model18.log_probs(m19, x19), "p26": model18.log_probs(m26, x26)})
        print(f"  {match_id}: done ({time.time() - start:.0f} s)")

    head = f"  {'':<34}{'spellF1':>8}{'macroF1':>9}{'accuracy':>10}{'poss':>7}{'found 1s':>10}{'correct':>9}"
    print("\n2. ABLATION (v19's model; hard overrides as in v25)")
    print(head)
    none = dict(P.DEFAULTS, w_out=0.0, w_set=0.0, w_goal=0.0, cleanup=0)
    steps = [("model alone (v19)", none),
             ("+ Law 9 ball out", dict(none, w_out=P.HARD)),
             ("+ set-piece windows", dict(none, w_set=P.HARD)),
             ("+ goal windows", dict(none, w_goal=P.HARD)),
             ("+ cleanup", dict(none, cleanup=1)),
             ("all of them (v25)", dict(P.DEFAULTS))]
    for name, p in steps:
        print(line(name, evaluate(folds, "p19", p)))
    print(f"  ({time.time() - start:.0f} s)")

    print("\n3. TUNING v26 (v26 features; all 7 matches)")
    best = tune(folds, "p26", P.DEFAULTS, log=print)
    print(head)
    print(line("v26 model, v25's settings", evaluate(folds, "p26", P.DEFAULTS)))
    print(line("v26 model, tuned settings", evaluate(folds, "p26", best)))
    print(f"  settings: {best}  ({time.time() - start:.0f} s)")

    print("\n4. FAIR EXTERNAL SCORE (settings tuned without the match being scored)")
    fair = []
    for k, f in enumerate(folds):
        rest = folds[:k] + folds[k + 1:]
        p = tune(rest, "p26", P.DEFAULTS, log=lambda *_: None)
        fair.append(evaluate([f], "p26", p)[0])
        print(f"  {f['id']}: spellF1 {fair[-1]['spell']:.3f} (v25 settings on v19's model: "
              f"{evaluate([f], 'p19', P.DEFAULTS)[0]['spell']:.3f})")
    print(head)
    print(line("v26 (fair)", fair))
    print(line("v25 on the same footing", evaluate(folds, "p19", P.DEFAULTS)))
    print(f"  ({time.time() - start:.0f} s)")

    print("\n5. FINAL MODEL (all 7 matches)")
    _, X26, y, names = rows(dfl.MATCHES, rng)
    model = model18.new_model().fit(X26, y)
    joblib.dump(model, MODEL_PATH)
    with open(PARAMS_PATH, "w") as handle:
        json.dump({"settings": {k: (float(v) if isinstance(v, float) else v) for k, v in best.items()},
                   "features": names, "cap_s": CAP_S, "fair_spell_f1": mean(fair),
                   "tuned_spell_f1": mean(evaluate(folds, "p26", best))}, handle, indent=2)
    print(f"  trained on {len(y):,} frames, {len(names)} features; saved "
          f"{os.path.relpath(MODEL_PATH, ROOT)} and {os.path.relpath(PARAMS_PATH, ROOT)}  "
          f"({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
