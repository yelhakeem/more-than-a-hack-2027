"""Locate the restart inside each Smart Tagging set-piece window, as the Laws define it.

    throw-in  (Law 15): the ball re-enters the pitch over a touchline
    corner    (Law 17): a still ball at a corner, then struck
    goal kick (Law 16): a still ball in a goal area, then struck
    free kick (Law 13): a still ball, then struck
    kick-off  (Law 8):  a still ball on the centre spot, then struck

Candidates are searched from BEFORE_TAG_S before the tag to AFTER_TAG_S after it; the one
closest to the default estimate is kept. When none is found, the restart falls back to the
median detected position for that type in this match (35 s windows) or RESTART_FRACTION of the
window (kick-offs).
"""

import numpy as np

from possession import config
from possession.tags import norm

FPS = config.FPS
STRUCK_MS = 5.0
STILL_WINDOW_S, STILL_GAP_S, STILL_RADIUS_M, STILL_MIN_SEEN = 0.5, 0.12, 0.5, 8
NEAR_PITCH_M = 1.5
BACK_ON_MIN_S = 0.3
CORNER_TOL_M, CORNER_OVER_M = 2.5, 1.5        # corner arc (1 m) plus tracking tolerance
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M, AREA_TOL_M = 5.5, 9.16, 1.5
CENTRE_TOL_M = 2.0
MIN_FOR_TYPE_MEDIAN = 3


def _strikes(t: dict, speed: np.ndarray) -> np.ndarray:
    """Frames where a ball near the pitch is struck after resting within STILL_RADIUS_M."""
    length, width = (float(v) for v in t["pitch"])
    x, y = t["bx"], t["by"]
    near = (~np.isnan(x) & (x > -NEAR_PITCH_M) & (x < length + NEAR_PITCH_M)
            & (y > -NEAR_PITCH_M) & (y < width + NEAR_PITCH_M))
    fast = near & (np.nan_to_num(speed) > STRUCK_MS)
    first = np.flatnonzero(fast & ~np.concatenate(([False], fast[:-1])))
    window, gap = int(STILL_WINDOW_S * FPS), int(STILL_GAP_S * FPS)
    out = []
    for i in first:
        a, b = i - gap - window, i - gap
        if a < 0:
            continue
        xs, ys = x[a:b], y[a:b]
        ok = ~np.isnan(xs)
        if ok.sum() < STILL_MIN_SEEN:
            continue
        cx, cy = np.median(xs[ok]), np.median(ys[ok])
        if np.max(np.hypot(xs[ok] - cx, ys[ok] - cy)) <= STILL_RADIUS_M:
            out.append(i)
    return np.array(out, dtype=np.int64)


def _back_on(t: dict) -> np.ndarray:
    """Frames where the ball, last seen beyond a touchline, is back on the pitch to stay."""
    width = float(t["pitch"][1])
    y = t["by"]
    seen = ~np.isnan(y)
    beyond = seen & ((y < 0) | (y > width))
    on_pitch = seen & ~beyond
    idx = np.where(seen, np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    prev_seen = np.concatenate(([-1], idx[:-1]))
    was_beyond = (prev_seen >= 0) & beyond[np.maximum(prev_seen, 0)]
    m = int(BACK_ON_MIN_S * FPS)
    stays = np.convolve(on_pitch.astype(int), np.ones(m, dtype=int), mode="full")[m - 1:] >= m
    return np.flatnonzero(on_pitch & was_beyond & stays)


def _in_position(t: dict, frame: int, kind: str) -> bool:
    """Whether the ball, in the second before `frame`, sat where this restart is taken."""
    length, width = (float(v) for v in t["pitch"])
    xs, ys = t["bx"][max(0, frame - FPS):frame], t["by"][max(0, frame - FPS):frame]
    ok = ~np.isnan(xs)
    if not ok.any():
        return False
    x, y = float(np.median(xs[ok])), float(np.median(ys[ok]))
    if kind == "corner kick":
        dx, dy = min(x, length - x), min(y, width - y)
        return -CORNER_OVER_M <= dx <= CORNER_TOL_M and -CORNER_OVER_M <= dy <= CORNER_TOL_M
    if kind == "goal kick":
        return (min(x, length - x) <= GOAL_AREA_DEPTH_M + AREA_TOL_M
                and abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH_M + AREA_TOL_M)
    if kind.startswith("kick off"):
        return np.hypot(x - length / 2, y - width / 2) <= CENTRE_TOL_M
    return True


def find_restarts(t: dict, speed: np.ndarray, tags: list) -> dict:
    """Return {id(tag): (restart_frame, detected)} for every set-piece tag."""
    n = len(t["bx"])
    strikes = _strikes(t, speed)
    back_on = _back_on(t)
    sets = [tag for tag in tags if norm(tag["code"]) == "set pieces"]
    out, offsets = {}, {}
    for tag in sets:
        kind = norm(tag["tags"].get("Type"))
        start, end = tag["start_frame"], tag["end_frame"]
        estimate = start + int(round(config.RESTART_FRACTION * (end - start)))
        a = max(0, start - int(config.BEFORE_TAG_S * FPS))
        b = min(n, end + int(config.AFTER_TAG_S * FPS))
        pool = back_on if kind == "throw-in" else strikes
        cands = pool[(pool >= a) & (pool < b)]
        if kind != "throw-in":
            cands = np.array([c for c in cands if _in_position(t, int(c), kind)], dtype=np.int64)
        if cands.size:
            frame = int(cands[np.argmin(np.abs(cands - estimate))])
            out[id(tag)] = (frame, True)
            if not kind.startswith("kick off"):
                offsets.setdefault(kind, []).append(frame - start)
        else:
            out[id(tag)] = None
    pooled = [v for values in offsets.values() for v in values]
    for tag in sets:
        if out[id(tag)] is not None:
            continue
        kind = norm(tag["tags"].get("Type"))
        start, end = tag["start_frame"], tag["end_frame"]
        if kind.startswith("kick off") or not pooled:
            frame = start + int(round(config.RESTART_FRACTION * (end - start)))
        else:
            values = offsets.get(kind, [])
            frame = start + int(np.median(values if len(values) >= MIN_FOR_TYPE_MEDIAN else pooled))
        out[id(tag)] = (frame, False)
    return out
