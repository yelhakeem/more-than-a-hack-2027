"""v27's second-stage features: the first model's answers over time, plus key clues.

For each frame: the first model's log-probabilities, the average of its probabilities over the
previous and the next 1, 3, 10 and 30 s and a centred 5 s, and a few clues from features26
(ball seen, ball out, set-piece window, restart timing, goals). Computed within each half.
"""

import numpy as np

FPS = 25
WINDOWS_S = (1.0, 3.0, 10.0, 30.0)
CENTRED_S = 5.0
CLUES = ("ball_seen", "ball_off", "since_seen", "until_seen", "set_in", "restart_since",
         "restart_until", "restart_detected", "off_run_s", "since_on_pitch", "goal_until")


def _window_means(p: np.ndarray, k: int) -> tuple:
    """Mean of the previous k and the next k rows (excluding the row itself), per column."""
    n = len(p)
    cs = np.vstack((np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)))
    idx = np.arange(n)
    a, b = np.maximum(idx - k, 0), idx
    past = (cs[b] - cs[a]) / np.maximum(b - a, 1)[:, None]
    past[idx == 0] = p[0]
    a, b = idx + 1, np.minimum(idx + 1 + k, n)
    future = (cs[b] - cs[a]) / np.maximum(b - a, 1)[:, None]
    future[idx == n - 1] = p[-1]
    return past, future


def build(log_probs: np.ndarray, X26: np.ndarray, names26: list, segments) -> tuple:
    """Second-stage feature matrix (frames x features) and names."""
    n = len(log_probs)
    p = np.exp(log_probs)
    blocks, names = [log_probs], ["lp_H", "lp_A", "lp_D"]
    cols = []
    for seconds in WINDOWS_S:
        cols += [f"past{int(seconds)}_{s}" for s in "HAD"] + [f"next{int(seconds)}_{s}" for s in "HAD"]
    cols += [f"mid{int(CENTRED_S)}_{s}" for s in "HAD"]
    out = np.full((n, len(cols)), np.nan, dtype=np.float32)
    for a, b in segments:
        part = p[a:b]
        pieces = []
        for seconds in WINDOWS_S:
            past, future = _window_means(part, int(seconds * FPS))
            pieces += [past, future]
        k = int(CENTRED_S * FPS) // 2
        past, future = _window_means(part, k)
        pieces.append((past + future + part) / 3)
        out[a:b] = np.hstack(pieces)
    blocks.append(out)
    names += cols
    idx = [names26.index(c) for c in CLUES]
    blocks.append(X26[:, idx])
    names += list(CLUES)
    return np.hstack([np.asarray(b, dtype=np.float32) for b in blocks]), names
