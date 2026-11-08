"""Per-frame features for the v18 model. Works on our match and on the external DFL matches.

Everything is seen from the home team's side: x is measured towards the goal the home team
attacks, so the model learns one picture of "home attacking" that holds in both halves and in
every match. Missing values stay NaN; the model handles them itself.

Input dict keys: px, py (frames x players, metres), bx, by (frames), team (players), pitch.
`segments` is a list of (start, stop) frame ranges played in one direction (the halves); the
first second of each must be the kick-off, when each team stands in its own half.
"""

import warnings

import numpy as np

FPS = 25
HOME = 0
RUNNING_MS, WALKING_MS = 4.0, 1.5     # standard tracking speed zones
NEAR_M = 5.0
OFF_MARGIN_M = 0.5
CAP_S = 30.0


def _nan_rolling(x: np.ndarray, seconds: float) -> np.ndarray:
    """Centred moving average that ignores NaN (NaN where the whole window is missing)."""
    k = max(1, int(seconds * FPS))
    ok = ~np.isnan(x)
    kernel = np.ones(k)
    total = np.convolve(np.where(ok, x, 0.0), kernel, mode="same")
    count = np.convolve(ok.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _shift_mean(x: np.ndarray, seconds: float, future: bool) -> np.ndarray:
    """Mean of the previous (or next) `seconds`, ignoring NaN."""
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
    idx = np.where(valid, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def _since(valid: np.ndarray, cap: int) -> np.ndarray:
    idx = np.where(valid, np.arange(len(valid)), -10 ** 9)
    np.maximum.accumulate(idx, out=idx)
    return np.minimum(np.arange(len(valid)) - idx, cap).astype(float)


def _player_speed(px: np.ndarray, py: np.ndarray, span: int = 2) -> np.ndarray:
    s = np.full(px.shape, np.nan, dtype=np.float32)
    s[span:-span] = np.hypot(px[2 * span:] - px[:-2 * span], py[2 * span:] - py[:-2 * span]) * FPS / (2 * span)
    return s


def _ball_speed(bx: np.ndarray, by: np.ndarray, span: int = 2) -> np.ndarray:
    s = np.full(len(bx), np.nan)
    s[span:-span] = np.hypot(bx[2 * span:] - bx[:-2 * span], by[2 * span:] - by[:-2 * span]) * FPS / (2 * span)
    return s


def _segment_block(t: dict, a: int, b: int, cap_s: float = CAP_S) -> dict:
    """All features for one half, from the home team's side."""
    length, width = (float(v) for v in t["pitch"])
    home = t["team"] == HOME
    px, py = t["px"][a:b].astype(np.float64), t["py"][a:b].astype(np.float64)
    bx, by = t["bx"][a:b].astype(np.float64), t["by"][a:b].astype(np.float64)
    n = b - a
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        k0 = px[:FPS]
        flip = np.nanmean(k0[:, home]) > np.nanmean(k0[:, ~home])   # home defends the high-x end
        if flip:
            px, bx = length - px, length - bx
        speed = _player_speed(px, py)
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
            ax = x if name == "home" else length - x          # away attacks towards x = 0
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
            fwd = np.full(n, np.nan)
            fwd[FPS:-FPS] = (mean_x[2 * FPS:] - mean_x[:-2 * FPS]) / 2
            f[f"{name}_forward"] = fwd
        f["push_diff"] = f["home_forward"] - f["away_forward"]
        f["height_diff"] = f["home_mean_x"] - (length - f["away_mean_x"])
        f["spread_ratio"] = np.log((f["home_spread"] + 1) / (f["away_spread"] + 1))
        allspeed = np.nanmean(speed, axis=1)
        f["all_speed"] = allspeed
        f["speed_drop"] = _shift_mean(allspeed, 1.0, False) - _shift_mean(allspeed, 1.0, True)

        for key in ("d_diff", "d_home", "d_away", "ball_speed", "home_speed", "away_speed",
                    "home_running", "away_running", "push_diff", "spread_ratio", "all_speed"):
            f[f"{key}_1s"] = _nan_rolling(f[key], 1.0)
            f[f"{key}_3s"] = _nan_rolling(f[key], 3.0)
        for key in ("ball_seen", "ball_off", "all_speed", "d_diff"):
            f[f"{key}_past3"] = _shift_mean(f[key], 3.0, False)
            f[f"{key}_next3"] = _shift_mean(f[key], 3.0, True)
    return f


def build(t: dict, segments, cap_s: float = CAP_S) -> tuple:
    """Feature matrix (frames x features, NaN outside the segments) and feature names."""
    n = len(t["bx"])
    blocks = [(a, b, _segment_block(t, a, b, cap_s)) for a, b in segments]
    names = list(blocks[0][2])
    X = np.full((n, len(names)), np.nan, dtype=np.float32)
    for a, b, f in blocks:
        X[a:b] = np.column_stack([np.asarray(f[k], dtype=np.float32) for k in names])
    return X, names
