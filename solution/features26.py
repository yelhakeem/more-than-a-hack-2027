"""v26 model features: v19's features plus Smart Tagging, the Laws' restarts and longer context.

Added on top of features18 (all seen from the home team's side, NaN-safe):
  - set-piece windows (Smart Tagging): inside a window, its type, the team named on it,
    seconds since it started and until it ends;
  - the restart found by restarts22 (Laws of the Game): seconds since / until the nearest one,
    and whether it was detected or estimated;
  - goals: seconds until / since a "Kick Off (After Goal)";
  - Law 9: how long the ball has been off the pitch, and since it was last seen on it;
  - 5 s and 10 s averages of the main movement features.
On the external DFL matches the tags are stand-ins built from DFL event data (external_eval.py).
"""

import numpy as np

import features18

FPS = 25
CAP_S = 60.0
LONG_S = (5.0, 10.0)
KINDS = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)")
SIDE = {"barcelona": 1.0, "athletic": -1.0, "atletic": -1.0}
ROLL_KEYS = ("home_speed", "away_speed", "all_speed", "home_running", "away_running",
             "ball_seen", "d_diff", "push_diff")


def _norm(value) -> str:
    return " ".join(str(value or "").lower().split())


def _side(team) -> float:
    team = _norm(team)
    return next((v for k, v in SIDE.items() if k in team), 0.0)


def _distance_to(points: np.ndarray, n: int, cap: int) -> tuple:
    """Frames since the last point and until the next point, capped."""
    since = np.full(n, cap, dtype=float)
    until = np.full(n, cap, dtype=float)
    if points.size:
        idx = np.arange(n)
        pos = np.searchsorted(points, idx, side="right")
        prev = np.where(pos > 0, points[np.maximum(pos - 1, 0)], -10 ** 9)
        nxt = np.where(pos < points.size, points[np.minimum(pos, points.size - 1)], 10 ** 9)
        since = np.minimum(idx - prev, cap).astype(float)
        until = np.minimum(nxt - idx, cap).astype(float)
    return since, until


def extra(t: dict, tags: list, found: dict, off: np.ndarray) -> dict:
    n = len(t["bx"])
    cap = int(CAP_S * FPS)
    f = {"set_in": np.zeros(n), "set_team": np.zeros(n),
         "set_since_start": np.full(n, CAP_S), "set_until_end": np.full(n, CAP_S)}
    for kind in KINDS:
        f["set_" + kind.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")] = np.zeros(n)
    restarts, detected, goals = [], [], []
    for tag in tags:
        if _norm(tag["code"]) != "set pieces":
            continue
        kind = _norm(tag["tags"].get("Type"))
        a, b = max(0, tag["start_frame"]), min(n, tag["end_frame"])
        if kind in KINDS and b > a:
            f["set_in"][a:b] = 1.0
            f["set_" + kind.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")][a:b] = 1.0
            f["set_team"][a:b] = _side(tag["tags"].get("Team"))
            f["set_since_start"][a:b] = np.arange(b - a) / FPS
            f["set_until_end"][a:b] = (b - a - np.arange(b - a)) / FPS
        if id(tag) in found and kind in KINDS:
            restarts.append(found[id(tag)][0])
            detected.append(found[id(tag)][1])
        if kind == "kick off (after goal)":
            goals.append(tag["start_frame"])
    order = np.argsort(restarts) if restarts else np.array([], dtype=int)
    points = np.array(restarts, dtype=np.int64)[order] if restarts else np.array([], dtype=np.int64)
    since, until = _distance_to(points, n, cap)
    f["restart_since"], f["restart_until"] = since / FPS, until / FPS
    if points.size:
        det = np.array(detected, dtype=float)[order]
        near = np.searchsorted(points, np.arange(n))
        near = np.clip(near, 0, points.size - 1)
        f["restart_detected"] = det[near]
    else:
        f["restart_detected"] = np.zeros(n)
    g_since, g_until = _distance_to(np.array(sorted(goals), dtype=np.int64), n, 3 * cap)
    f["goal_since"], f["goal_until"] = g_since / FPS, g_until / FPS
    seen_on = ~np.isnan(t["bx"]) & ~off
    run = np.zeros(n)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if off[i] else 0
    run[0] = 1 if off[0] else 0
    f["off_run_s"] = run / FPS
    idx = np.where(seen_on, np.arange(n), -10 ** 9)
    np.maximum.accumulate(idx, out=idx)
    f["since_on_pitch"] = np.minimum(np.arange(n) - idx, cap) / FPS
    return f


def build(t: dict, segments, tags: list, found: dict, off: np.ndarray, cap_s: float) -> tuple:
    """features18 (cap_s) + the extras above + long rolling averages. Returns X, names."""
    X, names = features18.build(t, segments, cap_s)
    ext = extra(t, tags, found, off)
    cols = [X] + [np.asarray(v, dtype=np.float32)[:, None] for v in ext.values()]
    names = names + list(ext)
    for key in ROLL_KEYS:
        base = X[:, names.index(key)].astype(np.float64)
        for seconds in LONG_S:
            col = np.full(len(base), np.nan)
            for a, b in segments:
                col[a:b] = features18._nan_rolling(base[a:b], seconds)
            cols.append(col.astype(np.float32)[:, None])
            names.append(f"{key}_{int(seconds)}s")
    return np.hstack(cols), names
