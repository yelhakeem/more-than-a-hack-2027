"""Shared pieces of v18: the classifier, the decoder and the external-data scorer."""

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

SEED = 0
PROB_FLOOR = 1e-3


def new_model() -> HistGradientBoostingClassifier:
    """Standard settings; nothing here is tuned on our match."""
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=SEED)


def log_probs(model, X: np.ndarray) -> np.ndarray:
    """Per-frame log probabilities for H, A, D (always 3 columns)."""
    p = np.full((len(X), 3), PROB_FLOOR)
    p[:, np.asarray(model.classes_, dtype=np.int64)] = model.predict_proba(X)
    p = np.maximum(p, PROB_FLOOR)
    return np.log(p / p.sum(axis=1, keepdims=True))


def viterbi(score: np.ndarray, switch_fee: float, dead_fee: float) -> np.ndarray:
    """Best H/A/D path: total per-frame score minus a fee for every change of state."""
    rows = score.tolist()
    n = len(rows)
    back = np.zeros((n, 3), dtype=np.int8)
    h, a, d = rows[0]
    for i in range(1, n):
        sh, sa, sd = rows[i]
        # into H
        ch = (h, 0) if h >= a - switch_fee and h >= d - dead_fee else ((a - switch_fee, 1) if a - switch_fee >= d - dead_fee else (d - dead_fee, 2))
        ca = (a, 1) if a >= h - switch_fee and a >= d - dead_fee else ((h - switch_fee, 0) if h - switch_fee >= d - dead_fee else (d - dead_fee, 2))
        cd = (d, 2) if d >= h - dead_fee and d >= a - dead_fee else ((h - dead_fee, 0) if h >= a else (a - dead_fee, 1))
        back[i, 0], back[i, 1], back[i, 2] = ch[1], ca[1], cd[1]
        h, a, d = ch[0] + sh, ca[0] + sa, cd[0] + sd
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax([h, a, d]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(score: np.ndarray, segments, switch_fee: float, dead_fee: float) -> np.ndarray:
    """Decode each segment separately; frames outside segments are -1."""
    out = np.full(len(score), -1, dtype=np.int8)
    for a, b in segments:
        out[a:b] = viterbi(score[a:b], switch_fee, dead_fee)
    return out


def score_external(labels: np.ndarray, pred: np.ndarray, segments) -> dict:
    """The competition's own metrics, on an external match (one X frame between halves)."""
    gt_parts, pr_parts = [], []
    for a, b in segments:
        gt_parts += [labels[a:b], np.array([pm.X], dtype=np.int8)]
        pr_parts += [pred[a:b], np.array([pm.X], dtype=np.int8)]
    gt, pr = np.concatenate(gt_parts[:-1]), np.concatenate(pr_parts[:-1])
    r = pm.evaluate(gt.astype(np.int8), pr.astype(np.int8))
    return {k: r[k] for k in ("spell_f1_mean", "macro_f1", "accuracy", "n_gt_spells", "n_pred_spells")}
