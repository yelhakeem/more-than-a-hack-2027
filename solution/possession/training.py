"""Training, tuning and scoring on the external DFL matches.

Scores use the organiser's own metric implementation (evaluation/possession_metrics.py),
applied to the DFL labels; the provided match's ground truth is never read.
"""

import numpy as np

import possession_metrics as pm
from possession import config, decode, external, model
from possession.pipeline import prepare

SUBSAMPLE = 5                          # neighbouring frames are near-duplicates
DEFAULT_SETTINGS = dict(switch_fee=8, dead_fee=48, w_out=decode.HARD, w_set=decode.HARD,
                        set_lead_s=12.0, w_goal=decode.HARD, cleanup=1)
GRID = {
    "switch_fee": (4, 6, 8, 12, 16),
    "dead_fee": (8, 16, 24, 32, 48),
    "w_set": (0.0, 0.5, 1.0, 2.0, decode.HARD),
    "set_lead_s": (2.0, 4.0, 8.0, 12.0),
    "w_goal": (0.0, decode.HARD),
    "cleanup": (0, 1),
}
PASSES = 2
TOLERANCE_FRAMES = 25                  # 1 s, the scorer's loose tolerance


def view_inputs(t: dict, match_id: str, swapped: bool = False):
    """Features and evidence for one (possibly team-swapped) degraded view of a match."""
    segments = [tuple(int(v) for v in s) for s in t["segments"]]
    tags = external.standin_tags(match_id, segments, swapped)
    return prepare(t, segments, tags, config.TIME_CAP_S)


def training_rows(match_ids, rng: np.random.Generator) -> tuple:
    """Subsampled rows from two degraded views (as is, and team-swapped) of each match."""
    Xs, ys, names = [], [], None
    for match_id in match_ids:
        clean = external.load_match(match_id)
        for swapped, view in ((False, clean), (True, external.swap_teams(clean))):
            t = external.degrade(view, rng)
            inputs = view_inputs(t, match_id, swapped)
            Xs.append(inputs.X[::SUBSAMPLE])
            ys.append(t["label"][::SUBSAMPLE])
            names = inputs.names
    return np.concatenate(Xs), np.concatenate(ys), names


def held_out(match_id: str, clf, missing_share: float = external.BALL_MISSING_SHARE,
             gap_scale: float = 1.0) -> dict:
    """Log-probabilities, truth and evidence for a match's fixed evaluation copy."""
    t = external.test_copy(match_id, missing_share, gap_scale)
    inputs = view_inputs(t, match_id)
    return {"id": match_id, "truth": t["label"], "segments": inputs.segments,
            "evidence": inputs.evidence, "lp": model.log_probs(clf, inputs.X)}


def _joined(labels: np.ndarray, pred: np.ndarray, segments) -> tuple:
    """Halves joined with one ungraded frame between them, as the scorer expects."""
    gt, pr = [], []
    for a, b in segments:
        gt += [labels[a:b], np.array([pm.X], dtype=np.int8)]
        pr += [pred[a:b], np.array([pm.X], dtype=np.int8)]
    return np.concatenate(gt[:-1]).astype(np.int8), np.concatenate(pr[:-1]).astype(np.int8)


def team_changes(codes: np.ndarray, runs) -> list:
    """[(frame, from_team, to_team)] each time possession passes to the other team."""
    out = []
    for start, stop in runs:
        last = None
        for frame, _, to in pm.transitions(codes[start:stop], [(0, stop - start)]):
            if to in (pm.H, pm.A):
                if last is not None and to != last:
                    out.append((start + frame, last, to))
                last = to
    return out


def score(truth: np.ndarray, codes: np.ndarray, segments) -> dict:
    """spellF1, macroF1, accuracy, possession counts and change-of-possession detection."""
    gt, pr = _joined(truth, codes, segments)
    r = pm.evaluate(gt, pr)
    runs = pm.valid_runs(gt)
    pairs, missed, spurious = pm.match_transitions(team_changes(gt, runs), team_changes(pr, runs),
                                                   TOLERANCE_FRAMES, typed="to")
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "acc": r["accuracy"],
            "poss": r["n_pred_spells"], "true": r["n_gt_spells"],
            "found": len(pairs) / max(1, len(pairs) + len(missed)),
            "correct": len(pairs) / max(1, len(pairs) + len(spurious))}


def evaluate(folds: list, settings: dict) -> list:
    return [score(f["truth"], decode.label(f["lp"], f["evidence"], f["segments"], settings), f["segments"])
            for f in folds]


def mean(results: list, key: str = "spell") -> float:
    return float(np.mean([r[key] for r in results]))


def tune(folds: list, start: dict, log=None) -> dict:
    """Coordinate search over GRID maximising mean spellF1 (Law 9 stays HARD)."""
    best = dict(start)
    best_score = mean(evaluate(folds, best))
    for _ in range(PASSES):
        changed = False
        for key, values in GRID.items():
            for value in values:
                if value == best[key]:
                    continue
                trial = dict(best, **{key: value})
                s = mean(evaluate(folds, trial))
                if s > best_score + 1e-4:
                    best, best_score, changed = trial, s, True
            if log:
                log(f"    {key:<11} -> {best[key]!s:<7} (spellF1 {best_score:.3f})")
        if not changed:
            break
    return best


def summary_row(name: str, results: list) -> str:
    return (f"  {name:<34}{mean(results):>8.3f}{mean(results, 'macro'):>9.3f}"
            f"{100 * mean(results, 'acc'):>9.1f}%{mean(results, 'poss'):>7.0f}"
            f"{100 * mean(results, 'found'):>9.1f}%{100 * mean(results, 'correct'):>9.1f}%")


SUMMARY_HEADER = (f"  {'':<34}{'spellF1':>8}{'macroF1':>9}{'accuracy':>10}{'poss':>7}"
                  f"{'found 1s':>10}{'correct':>9}")
