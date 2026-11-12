"""Per-frame feature matrix used by the classifier.

The same code runs on the provided match and on the external training matches. Geometry is
expressed from the home team's side (x increases towards the goal the home team attacks), so
one model serves both halves and every match. Missing values stay NaN; the classifier handles
them natively.

Feature groups:
    ball        position, speed, distance to each team, time since/until last seen or off pitch
    teams       shape (depth, width, spread), speed, running/walking counts, forward movement
    context     1 s / 3 s / 5 s / 10 s rolling averages and 3 s past/future means
    tagging     Smart Tagging set-piece windows (type, team, time in window), the restart found
                by possession.restarts, goal kick-offs
    law 9       how long the ball has been off the pitch and since it was last seen on it
"""

import warnings

import numpy as np

from possession import config
from possession.tags import norm

FPS = config.FPS
RUNNING_MS, WALKING_MS = 4.0, 1.5      # standard tracking speed zones
NEAR_M = 5.0
OFF_MARGIN_M = 0.5
SHORT_WINDOWS_S = (1.0, 3.0)
LONG_WINDOWS_S = (5.0, 10.0)
TAG_KINDS = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)")
TEAM_SIDE = {"barcelona": 1.0, "athletic": -1.0, "atletic": -1.0}
LONG_ROLL_KEYS = ("home_speed", "away_speed", "all_speed", "home_running", "away_running",
                  "ball_seen", "d_diff", "push_diff")


# --- Generic helpers ---------------------------------------------------------------------------

def nan_rolling(x: np.ndarray, seconds: float) -> np.ndarray:
    """Centred moving average ignoring NaN (NaN where the whole window is missing)."""
    k = max(1, int(seconds * FPS))
    ok = ~np.isnan(x)
    kernel = np.ones(k)
    total = np.convolve(np.where(ok, x, 0.0), kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _shift_mean(x: np.ndarray, seconds: float, future: bool) -> np.ndarray:
    """Mean over the previous (or next) `seconds`, ignoring NaN."""
    k = max(1, int(seconds * FPS))
    ok = ~np.isnan(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cc = np.concatenate(([0.0], np.cumsum(ok)))
    idx = np.arange(len(x))
    if future:
        a, b = idx + 1, np.minimum(idx + 1 + k, len(x))
    else:
        a, b = np.maximum(idx - k, 0), idx
    total, count = cs[b] - cs[a], cc[b] - cc[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _ffill(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Carry the last valid value forward (NaN before the first one)."""
    idx = np.where(valid, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def _since(valid: np.ndarray, cap: int) -> np.ndarray:
    """Frames since the last True, capped at `cap`."""
    idx = np.where(valid, np.arange(len(valid)), -10 ** 9)
    np.maximum.accumulate(idx, out=idx)
    return np.minimum(np.arange(len(valid)) - idx, cap).astype(float)


def player_speed(px: np.ndarray, py: np.ndarray, span: int = 2) -> np.ndarray:
    """Per-player speed in m/s (centred difference over 2 * span frames)."""
    s = np.full(px.shape, np.nan, dtype=np.float32)
    s[span:-span] = np.hypot(px[2 * span:] - px[:-2 * span], py[2 * span:] - py[:-2 * span]) * FPS / (2 * span)
    return s


def _ball_speed(bx: np.ndarray, by: np.ndarray, span: int = 2) -> np.ndarray:
    s = np.full(len(bx), np.nan)
    s[span:-span] = np.hypot(bx[2 * span:] - bx[:-2 * span], by[2 * span:] - by[:-2 * span]) * FPS / (2 * span)
    return s


def _distance_to(points: np.ndarray, n: int, cap: int) -> tuple:
    """Frames since the previous point and until the next one, capped."""
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


# --- Tracking features (one half at a time) ----------------------------------------------------

def _half_features(t: dict, a: int, b: int, cap_s: float) -> dict:
    """Ball, team and context features for frames [a, b) of one half."""
    length, width = (float(v) for v in t["pitch"])
    home = t["team"] == config.HOME
    px, py = t["px"][a:b].astype(np.float64), t["py"][a:b].astype(np.float64)
    bx, by = t["bx"][a:b].astype(np.float64), t["by"][a:b].astype(np.float64)
    n = b - a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Orientation: at kick-off each team stands in its own half.
        k0 = px[:FPS]
        if np.nanmean(k0[:, home]) > np.nanmean(k0[:, ~home]):
            px, bx = length - px, length - bx
        speed = player_speed(px, py)
        dist = np.hypot(px - bx[:, None], py - by[:, None])

        f = {}
        seen = ~np.isnan(bx)
        f["ball_seen"] = seen.astype(float)
        f["ball_x"] = bx
        f["ball_y_centre"] = np.abs(by - width / 2)
        off = (bx < -OFF_MARGIN_M) | (bx > length + OFF_MARGIN_M) | (by < -OFF_MARGIN_M) | (by > width + OFF_MARGIN_M)
        f["ball_off"] = np.where(seen, off.astype(float), np.nan)
        f["ball_speed"] = _ball_speed(bx, by)
        d_home, d_away = np.nanmin(dist[:, home], axis=1), np.nanmin(dist[:, ~home], axis=1)
        f["d_home"], f["d_away"] = d_home, d_away
        f["d_diff"] = d_home - d_away
        f["n_home_near"] = np.where(seen, np.sum(dist[:, home] <= NEAR_M, axis=1), np.nan)
        f["n_away_near"] = np.where(seen, np.sum(dist[:, ~home] <= NEAR_M, axis=1), np.nan)

        cap = int(cap_s * FPS)
        rev = slice(None, None, -1)
        f["since_seen"] = _since(seen, cap) / FPS
        f["until_seen"] = _since(seen[rev], cap)[rev] / FPS
        for key in ("d_diff", "ball_x", "ball_off", "ball_y_centre"):
            f[f"last_{key}"] = _ffill(f[key], seen)
            f[f"next_{key}"] = _ffill(f[key][rev], seen[rev])[rev]
        f["since_off"] = _since(np.nan_to_num(f["ball_off"]) > 0, cap) / FPS
        f["until_off"] = _since((np.nan_to_num(f["ball_off"]) > 0)[rev], cap)[rev] / FPS

        for name, mask in (("home", home), ("away", ~home)):
            x, y, s = px[:, mask], py[:, mask], speed[:, mask]
            ax = x if name == "home" else length - x          # each team in its attacking direction
            mean_x = np.nanmean(ax, axis=1)
            f[f"{name}_tracked"] = np.sum(~np.isnan(x), axis=1).astype(float)
            f[f"{name}_mean_x"] = mean_x
            f[f"{name}_front_x"] = np.nanmax(ax, axis=1)
            f[f"{name}_back_x"] = np.nanmin(ax, axis=1)
            f[f"{name}_width"] = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
            f[f"{name}_spread"] = np.nanstd(ax, axis=1) * np.nanstd(y, axis=1)
            f[f"{name}_speed"] = np.nanmean(s, axis=1)
            f[f"{name}_running"] = np.nansum(s > RUNNING_MS, axis=1).astype(float)
            f[f"{name}_walking"] = np.nansum(s < WALKING_MS, axis=1).astype(float)
            forward = np.full(n, np.nan)
            forward[FPS:-FPS] = (mean_x[2 * FPS:] - mean_x[:-2 * FPS]) / 2
            f[f"{name}_forward"] = forward
        f["push_diff"] = f["home_forward"] - f["away_forward"]
        f["height_diff"] = f["home_mean_x"] - (length - f["away_mean_x"])
        f["spread_ratio"] = np.log((f["home_spread"] + 1) / (f["away_spread"] + 1))
        all_speed = np.nanmean(speed, axis=1)
        f["all_speed"] = all_speed
        f["speed_drop"] = _shift_mean(all_speed, 1.0, False) - _shift_mean(all_speed, 1.0, True)

        for key in ("d_diff", "d_home", "d_away", "ball_speed", "home_speed", "away_speed",
                    "home_running", "away_running", "push_diff", "spread_ratio", "all_speed"):
            for seconds in SHORT_WINDOWS_S:
                f[f"{key}_{int(seconds)}s"] = nan_rolling(f[key], seconds)
        for key in ("ball_seen", "ball_off", "all_speed", "d_diff"):
            f[f"{key}_past3"] = _shift_mean(f[key], 3.0, False)
            f[f"{key}_next3"] = _shift_mean(f[key], 3.0, True)
    return f


def _tracking_matrix(t: dict, segments, cap_s: float) -> tuple:
    n = len(t["bx"])
    blocks = [(a, b, _half_features(t, a, b, cap_s)) for a, b in segments]
    names = list(blocks[0][2])
    X = np.full((n, len(names)), np.nan, dtype=np.float32)
    for a, b, f in blocks:
        X[a:b] = np.column_stack([np.asarray(f[k], dtype=np.float32) for k in names])
    return X, names


# --- Smart Tagging and Law 9 features ----------------------------------------------------------

def _tag_column(kind: str) -> str:
    return "set_" + kind.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def _team_side(team) -> float:
    team = " ".join(str(team or "").lower().split())
    return next((v for k, v in TEAM_SIDE.items() if k in team), 0.0)


def _context_features(t: dict, tags: list, restarts: dict, off: np.ndarray) -> dict:
    n = len(t["bx"])
    cap = int(config.TIME_CAP_S * FPS)
    f = {"set_in": np.zeros(n), "set_team": np.zeros(n),
         "set_since_start": np.full(n, config.TIME_CAP_S), "set_until_end": np.full(n, config.TIME_CAP_S)}
    for kind in TAG_KINDS:
        f[_tag_column(kind)] = np.zeros(n)
    frames, detected, goals = [], [], []
    for tag in tags:
        if " ".join(str(tag["code"] or "").lower().split()) != "set pieces":
            continue
        kind = " ".join(str(tag["tags"].get("Type") or "").lower().split())
        a, b = max(0, tag["start_frame"]), min(n, tag["end_frame"])
        if kind in TAG_KINDS and b > a:
            f["set_in"][a:b] = 1.0
            f[_tag_column(kind)][a:b] = 1.0
            f["set_team"][a:b] = _team_side(tag["tags"].get("Team"))
            f["set_since_start"][a:b] = np.arange(b - a) / FPS
            f["set_until_end"][a:b] = (b - a - np.arange(b - a)) / FPS
        if id(tag) in restarts and kind in TAG_KINDS:
            frames.append(restarts[id(tag)][0])
            detected.append(restarts[id(tag)][1])
        if kind == "kick off (after goal)":
            goals.append(tag["start_frame"])

    order = np.argsort(frames) if frames else np.array([], dtype=int)
    points = np.array(frames, dtype=np.int64)[order] if frames else np.array([], dtype=np.int64)
    since, until = _distance_to(points, n, cap)
    f["restart_since"], f["restart_until"] = since / FPS, until / FPS
    if points.size:
        flags = np.array(detected, dtype=float)[order]
        nearest = np.clip(np.searchsorted(points, np.arange(n)), 0, points.size - 1)
        f["restart_detected"] = flags[nearest]
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


# --- Public entry point ------------------------------------------------------------------------

def build(t: dict, segments, tags: list, restarts: dict, off: np.ndarray,
          cap_s: float = config.TIME_CAP_S) -> tuple:
    """Return the feature matrix (frames x features, float32) and the feature names.

    Args:
        t: tracking dict (px, py, bx, by, team, pitch).
        segments: [(start, stop)] of each half, starting at its kick-off.
        tags: Smart Tagging events (or external stand-ins).
        restarts: output of possession.restarts.find_restarts.
        off: ball-off-pitch mask (possession.signals.off_pitch).
        cap_s: cap for "time since / until" features.
    """
    X, names = _tracking_matrix(t, segments, cap_s)
    extra = _context_features(t, tags, restarts, off)
    columns = [X] + [np.asarray(v, dtype=np.float32)[:, None] for v in extra.values()]
    names = names + list(extra)
    for key in LONG_ROLL_KEYS:
        base = X[:, names.index(key)].astype(np.float64)
        for seconds in LONG_WINDOWS_S:
            col = np.full(len(base), np.nan)
            for a, b in segments:
                col[a:b] = nan_rolling(base[a:b], seconds)
            columns.append(col.astype(np.float32)[:, None])
            names.append(f"{key}_{int(seconds)}s")
    return np.hstack(columns), names
