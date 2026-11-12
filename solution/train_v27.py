"""Train v27 (v26's model + a second model that reads its answers over time) on the 7 external
DFL matches, and compare it fairly with v26.

  1. For every DFL match: two training views (as is, and seen from the other side), each
     degraded like our video, plus the held-out test copy (the same seed as before). Features
     are built once and cached in data/external/v27_cache/.
  2. First stage (v26's model): for each match, a model trained on the other six predicts all
     three copies of it, so every first-stage answer is out-of-fold.
  3. Second stage: for each match, a model trained on the other six matches' first-stage answers
     (stack27 features) predicts its test copy.
  4. Fair external score for v26 AND v27: decoder settings tuned with each match left out
     (tune_v26's search), scored on that match.
  5. Final models on all 7; saved to solution/models/ with the tuned settings.

Rule fixed before any result: v27 replaces v26 only if it passes the four label-free checks
on our match, its fair spellF1 beats v26's (from this run) by at least 0.01, and it is better
on at least 5 of the 7 matches. v27.py prints the decision.

External data only. Usage:
    python solution/train_v27.py      # ~1-1.5 hours, ~1.5 GB of cache on disk
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
import features26  # noqa: E402
import model18  # noqa: E402
import pipeline26 as P  # noqa: E402
import stack27  # noqa: E402
import train_v19  # noqa: E402
import tune_v26  # noqa: E402
import v2  # noqa: E402

CACHE = os.path.join(ROOT, "data", "external", "v27_cache")
STAGE1_PATH = os.path.join(train_v19.MODEL_DIR, "v27_stage1.joblib")
STAGE2_PATH = os.path.join(train_v19.MODEL_DIR, "v27_stage2.joblib")
PARAMS_PATH = os.path.join(train_v19.MODEL_DIR, "v27_params.json")
SUB = train_v19.SUBSAMPLE
VIEWS = ("train", "swap", "test")


def prepare_all(rng) -> dict:
    """Build and cache features for every match and view; return per-view metadata."""
    os.makedirs(CACHE, exist_ok=True)
    meta = {}
    for idx, match_id in enumerate(dfl.MATCHES):
        clean = dfl.load_match(match_id)
        sources = {"train": (dfl.degrade(clean, rng), False),
                   "swap": (dfl.degrade(dfl.swap_teams(clean), rng), True),
                   "test": (dfl.degrade(clean, np.random.default_rng(1000 + idx), train_v19.TEST_MISSING), False)}
        for view, (t, swapped) in sources.items():
            t, segments, tags, found, off = tune_v26.match_inputs(t, match_id, swapped)
            X, names = features26.build(t, segments, tags, found, off, train_v19.CAP_S)
            path = os.path.join(CACHE, f"{match_id}_{view}.npy")
            np.save(path, X)
            meta[(match_id, view)] = {"path": path, "label": t["label"], "segments": segments,
                                      "masks": P.masks(t, tags, found, off, v2._norm, v2.SET_PIECE_TYPES)}
        print(f"  {match_id}: features cached")
    return meta, names


def load(meta: dict, key) -> np.ndarray:
    return np.load(meta[key]["path"], mmap_mode="r")


def rows(meta: dict, matches, stage: str, stage1_lp: dict = None, names=None) -> tuple:
    """Training rows (every SUB-th frame) from the train and swap views of the given matches."""
    Xs, ys = [], []
    for m in matches:
        for view in ("train", "swap"):
            key = (m, view)
            X = load(meta, key)
            if stage == "1":
                Xs.append(np.asarray(X[::SUB]))
            else:
                S, _ = stack27.build(stage1_lp[key], X, names, meta[key]["segments"])
                Xs.append(S[::SUB])
            ys.append(meta[key]["label"][::SUB])
    return np.concatenate(Xs), np.concatenate(ys)


def fold(meta: dict, key: tuple, lp: np.ndarray) -> dict:
    return {"id": key[0], "truth": meta[key]["label"], "segments": meta[key]["segments"],
            "masks": meta[key]["masks"], "p": lp}


def fair(folds: list) -> list:
    out = []
    for k, f in enumerate(folds):
        rest = folds[:k] + folds[k + 1:]
        p = tune_v26.tune(rest, "p", P.DEFAULTS, log=lambda *_: None)
        out.append(tune_v26.evaluate([f], "p", p)[0])
    return out


def main() -> None:
    start = time.time()
    rng = np.random.default_rng(model18.SEED)
    print("1. FEATURES (external DFL data only)")
    meta, names = prepare_all(rng)
    print(f"  ({time.time() - start:.0f} s)")

    print("\n2. FIRST STAGE (out-of-fold)")
    lp1 = {}
    for m in dfl.MATCHES:
        X, y = rows(meta, [o for o in dfl.MATCHES if o != m], "1")
        model = model18.new_model().fit(X, y)
        for view in VIEWS:
            lp1[(m, view)] = model18.log_probs(model, np.asarray(load(meta, (m, view))))
        print(f"  {m}: done ({time.time() - start:.0f} s)")

    print("\n3. SECOND STAGE (out-of-fold)")
    lp2 = {}
    for m in dfl.MATCHES:
        X, y = rows(meta, [o for o in dfl.MATCHES if o != m], "2", lp1, names)
        model = model18.new_model().fit(X, y)
        key = (m, "test")
        S, _ = stack27.build(lp1[key], load(meta, key), names, meta[key]["segments"])
        lp2[key] = model18.log_probs(model, S)
        print(f"  {m}: done ({time.time() - start:.0f} s)")

    print("\n4. FAIR EXTERNAL SCORE (settings tuned without the match being scored)")
    f26 = [fold(meta, (m, "test"), lp1[(m, "test")]) for m in dfl.MATCHES]
    f27 = [fold(meta, (m, "test"), lp2[(m, "test")]) for m in dfl.MATCHES]
    r26, r27 = fair(f26), fair(f27)
    better = 0
    for m, a, b in zip(dfl.MATCHES, r26, r27):
        better += b["spell"] > a["spell"]
        print(f"  {m}: v26 {a['spell']:.3f}   v27 {b['spell']:.3f}")
    head = (f"  {'':<34}{'spellF1':>8}{'macroF1':>9}{'accuracy':>10}{'poss':>7}{'found 1s':>10}{'correct':>9}")
    print(head)
    print(tune_v26.line("v26 (fair, this run)", r26))
    print(tune_v26.line("v27 (fair)", r27))
    gain = tune_v26.mean(r27) - tune_v26.mean(r26)
    print(f"  gain {gain:+.3f}; better on {better} of {len(dfl.MATCHES)} matches  ({time.time() - start:.0f} s)")

    print("\n5. FINAL MODELS (all 7 matches)")
    best = tune_v26.tune(f27, "p", P.DEFAULTS, log=lambda *_: None)
    X, y = rows(meta, dfl.MATCHES, "1")
    joblib.dump(model18.new_model().fit(X, y), STAGE1_PATH)
    X, y = rows(meta, dfl.MATCHES, "2", lp1, names)
    joblib.dump(model18.new_model().fit(X, y), STAGE2_PATH)
    _, names2 = stack27.build(lp1[(dfl.MATCHES[0], "test")][:10], np.asarray(load(meta, (dfl.MATCHES[0], "test"))[:10]),
                              names, [(0, 10)])
    with open(PARAMS_PATH, "w") as handle:
        json.dump({"settings": {k: (float(v) if isinstance(v, float) else v) for k, v in best.items()},
                   "features": names, "stage2_features": names2, "cap_s": train_v19.CAP_S,
                   "fair_spell_f1": tune_v26.mean(r27), "fair_spell_f1_v26_this_run": tune_v26.mean(r26),
                   "gain": gain, "better_matches": int(better)}, handle, indent=2)
    print(f"  settings: {best}")
    print(f"  saved {os.path.relpath(STAGE1_PATH, ROOT)}, {os.path.relpath(STAGE2_PATH, ROOT)}, "
          f"{os.path.relpath(PARAMS_PATH, ROOT)}  ({time.time() - start:.0f} s)")


if __name__ == "__main__":
    main()
