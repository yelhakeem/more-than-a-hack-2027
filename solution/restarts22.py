"""v22's restart finder: restarts.py with a sturdier "still ball" test and a measured fallback.

Changes from restarts.py (v21):
  - Still ball: video tracking jitters, so speed computed frame to frame rarely drops below
    1 m/s even for a ball sitting on the spot. A ball now counts as still when, over the
    0.5 s before it is struck, it stays within 0.5 m of where it sits (at least 8 of those
    frames seen).
  - Struck near a line: a corner ball can be tracked just over the line, so a strike only
    needs the ball seen and within 1.5 m of the pitch, not strictly on it.
  - Fallback: Smart Tagging set-piece windows are all 35 s long, so "68% through" (measured
    on the 5 s kick-off windows) does not transfer. When nothing is detected, the restart is
    put at the median position of the restarts detected for that type in this match (or of
    all 35 s set pieces, if that type has fewer than 3), measured from the tag start.
    Kick-off windows keep 68%. This is an automatic check on the provided tracking.
"""

import numpy as np

import restarts as base

FPS = base.FPS
STILL_WINDOW_S, STILL_GAP_S, STILL_RADIUS_M, STILL_MIN_SEEN = 0.5, 0.12, 0.5, 8
NEAR_PITCH_M = 1.5
CORNER_OVER_M = 1.5
MIN_FOR_TYPE_MEDIAN = 3


def _strikes(t: dict, speed: np.ndarray) -> np.ndarray:
    """Frames where a ball near the pitch is struck after sitting still for 0.5 s."""
    length, width = (float(v) for v in t["pitch"])
    x, y = t["bx"], t["by"]
    near = (~np.isnan(x) & (x > -NEAR_PITCH_M) & (x < length + NEAR_PITCH_M)
            & (y > -NEAR_PITCH_M) & (y < width + NEAR_PITCH_M))
    fast = near & (np.nan_to_num(speed) > base.STRUCK_MS)
    first = np.flatnonzero(fast & ~np.concatenate(([False], fast[:-1])))
    w, gap = int(STILL_WINDOW_S * FPS), int(STILL_GAP_S * FPS)
    out = []
    for i in first:
        a, b = i - gap - w, i - gap
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


def _where(t: dict, frame: int, kind: str) -> bool:
    if kind != "corner kick":
        return base._where(t, frame, kind)
    length, width = (float(v) for v in t["pitch"])
    xs, ys = t["bx"][max(0, frame - FPS):frame], t["by"][max(0, frame - FPS):frame]
    ok = ~np.isnan(xs)
    if not ok.any():
        return False
    x, y = float(np.median(xs[ok])), float(np.median(ys[ok]))
    dx, dy = min(x, length - x), min(y, width - y)
    return -CORNER_OVER_M <= dx <= base.CORNER_TOL_M and -CORNER_OVER_M <= dy <= base.CORNER_TOL_M


def find_restarts(t: dict, speed: np.ndarray, off: np.ndarray, tags: list, norm) -> dict:
    """{id(tag): (frame, detected?)} for every set-piece tag."""
    n = len(t["bx"])
    strikes = _strikes(t, speed)
    back_on = base._back_on(t)
    sets = [tag for tag in tags if norm(tag["code"]) == "set pieces"]
    out, offsets = {}, {}
    for tag in sets:
        kind = norm(tag["tags"].get("Type"))
        start, end = tag["start_frame"], tag["end_frame"]
        estimate = start + int(round(base.RESTART_FRACTION * (end - start)))
        a, b = max(0, start - int(base.BEFORE_TAG_S * FPS)), min(n, end + int(base.AFTER_TAG_S * FPS))
        pool = back_on if kind == "throw-in" else strikes
        cands = pool[(pool >= a) & (pool < b)]
        if kind != "throw-in":
            cands = np.array([c for c in cands if _where(t, int(c), kind)], dtype=np.int64)
        if cands.size:
            frame = int(cands[np.argmin(np.abs(cands - estimate))])
            out[id(tag)] = (frame, True)
            if not kind.startswith("kick off"):
                offsets.setdefault(kind, []).append(frame - start)
        else:
            out[id(tag)] = None
    pooled = [v for vals in offsets.values() for v in vals]
    for tag in sets:
        if out[id(tag)] is not None:
            continue
        kind = norm(tag["tags"].get("Type"))
        start, end = tag["start_frame"], tag["end_frame"]
        if kind.startswith("kick off") or not pooled:
            frame = start + int(round(base.RESTART_FRACTION * (end - start)))
        else:
            vals = offsets.get(kind, [])
            frame = start + int(np.median(vals if len(vals) >= MIN_FOR_TYPE_MEDIAN else pooled))
        out[id(tag)] = (frame, False)
    return out
