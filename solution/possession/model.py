"""Frame classifier: gradient-boosted trees giving P(H), P(A), P(D) per frame."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 0
PROB_FLOOR = 1e-3


def new_model() -> HistGradientBoostingClassifier:
    """Return an untrained classifier with the fixed hyperparameters used throughout."""
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, early_stopping=False,
                                          random_state=SEED)


def log_probs(model, X: np.ndarray) -> np.ndarray:
    """Per-frame log-probabilities for H, A, D (always three columns, floored at PROB_FLOOR)."""
    p = np.full((len(X), 3), PROB_FLOOR)
    p[:, np.asarray(model.classes_, dtype=np.int64)] = model.predict_proba(X)
    p = np.maximum(p, PROB_FLOOR)
    return np.log(p / p.sum(axis=1, keepdims=True))
