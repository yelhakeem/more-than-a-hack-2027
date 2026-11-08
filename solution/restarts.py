"""Find the actual restart inside each Smart Tagging set-piece window, the way the Laws describe it.

  throw-in  (Law 15): the ball comes back onto the pitch over a touchline
  corner    (Law 17): a still ball at a corner, then struck
  goal kick (Law 16): a still ball in a goal area, then struck
  free kick (Law 13): a still ball, then struck
  kick-off  (Law 8):  a still ball on the centre spot, then struck

Search range: from 4 s before the tag to 3 s after it (v2's window). When several candidates
are found, the one closest to v4's estimate (68% through the tag) is used; when none is found,
the estimate is kept. Uses the tracking and the tags only.
"""

import numpy as np

FPS = 25
STILL_MS, STRUCK_MS = 1.0, 5.0
STILL_MIN_S, STRIKE_WITHIN_S = 0.4, 1.0
CORNER_TOL_M = 2.5                     # corner arc (1 m) + tracking tolerance
GOAL_AREA_DEPTH_M, GOAL_AREA_HALF_WIDTH_M, AREA_TOL_M = 5.5, 9.16, 1.5
CENTRE_TOL_M = 2.0
BACK_ON_MIN_S = 0.3
BEFORE_TAG_S, AFTER_TAG_S, RESTART_FRACTION = 4.0, 3.0, 0.68


def _strikes(speed: np.ndarray, on_pitch: np.ndarray) -> np.ndarray:
    """Frames where an on-pitch ball is struck after resting still for STILL_MIN_S."""
    k = int(STRIKE_WITHIN_S * FPS)
    still = np.nan_to_num(speed, nan=np.inf) < STILL_MS
    counts = np.convolve(still.astype(int), np.ones(k, dtype=int), mode="full")[:len(still)]
    before = np.concatenate(([0], counts[:-1]))          # still frames in the k frames before i
    fast = on_pitch & (np.nan_to_num(speed) > STRUCK_MS)
    first_fast = fast & ~np.concatenate(([False], fast[:-1]))
    return np.flatnonzero(first_fast & (before >= int(STILL_MIN_S * FPS)))


def _back_on(t: dict) -> np.ndarray:
    """Frames where the ball, last seen beyond a touchline, is seen back on the pitch to stay."""
    width = float(t["pitch"][1])
    y = t["by"]
    seen = ~np.isnan(y)
    beyond = seen & ((y < 0) | (y > width))
    on_pitch = seen & ~beyond                              # strictly inside the touchlines
    idx = np.where(seen, np.arange(len(y)), -1)
    np.maximum.accumulate(idx, out=idx)
    prev_seen = np.concatenate(([-1], idx[:-1]))
    was_beyond = (prev_seen >= 0) & beyond[np.maximum(prev_seen, 0)]
    m = int(BACK_ON_MIN_S * FPS)
    stays = np.convolve(on_pitch.astype(int), np.ones(m, dtype=int), mode="full")[m - 1:] >= m
    return np.flatnonzero(on_pitch & was_beyond & stays)


def _where(t: dict, frame: int, kind: str) -> bool:
    """Was the ball, just before `frame`, where this kind of restart is taken?"""
    length, width = (float(v) for v in t["pitch"])
    xs, ys = t["bx"][max(0, frame - FPS):frame], t["by"][max(0, frame - FPS):frame]
    ok = ~np.isnan(xs)
    if not ok.any():
        return False
    x, y = float(np.median(xs[ok])), float(np.median(ys[ok]))
    if kind == "corner kick":
        return min(x, length - x) <= CORNER_TOL_M and min(y, width - y) <= CORNER_TOL_M
    if kind == "goal kick":
        return (min(x, length - x) <= GOAL_AREA_DEPTH_M + AREA_TOL_M
                and abs(y - width / 2) <= GOAL_AREA_HALF_WIDTH_M + AREA_TOL_M)
    if kind.startswith("kick off"):
        return np.hypot(x - length / 2, y - width / 2) <= CENTRE_TOL_M
    return True                                            # free kick: anywhere


def find_restarts(t: dict, speed: np.ndarray, off: np.ndarray, tags: list, norm) -> dict:
    """{id(tag): (frame, detected?)} for every set-piece tag; norm() normalises tag strings."""
    n = len(t["bx"])
    on_pitch = ~np.isnan(t["bx"]) & ~off
    strikes = _strikes(speed, on_pitch)
    back_on = _back_on(t)
    out = {}
    for tag in tags:
        if norm(tag["code"]) != "set pieces":
            continue
        kind = norm(tag["tags"].get("Type"))
        start, end = tag["start_frame"], tag["end_frame"]
        estimate = start + int(round(RESTART_FRACTION * (end - start)))
        a, b = max(0, start - int(BEFORE_TAG_S * FPS)), min(n, end + int(AFTER_TAG_S * FPS))
        pool = back_on if kind == "throw-in" else strikes
        cands = pool[(pool >= a) & (pool < b)]
        if kind != "throw-in":
            cands = np.array([c for c in cands if _where(t, int(c), kind)], dtype=np.int64)
        if cands.size:
            out[id(tag)] = (int(cands[np.argmin(np.abs(cands - estimate))]), True)
        else:
            out[id(tag)] = (estimate, False)
    return out
