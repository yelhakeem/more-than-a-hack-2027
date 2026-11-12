"""Turn per-frame probabilities into a label sequence.

    score  = classifier log-probabilities
           + w_out  toward D where Law 9 says the ball is out of play
           + w_set  toward D for set_lead_s before each set-piece restart
           + w_goal toward D in the 41 s before each kick-off after a goal
    labels = Viterbi path over H / A / D with a fee for every change of state, then an
             optional clean-up of implausibly short stoppages and plays.

A weight of HARD forces the state. All settings are tuned on external data (train.py); Law 9
is a definition, not a parameter, and is always HARD.
"""

import numpy as np

from possession import config

H, A, D = config.H, config.A, config.D
FPS = config.FPS
HARD = 1000.0
GOAL_WINDOW_S = 41.0                   # goal delay mean - 1 SD (Augste & Cordes, 2016)
MIN_DEAD_S, MIN_PLAY_S = 3.0, 1.0


def viterbi(score: np.ndarray, switch_fee: float, dead_fee: float) -> np.ndarray:
    """Best H/A/D path: total per-frame score minus a fee for every change of state."""
    rows = score.tolist()
    n = len(rows)
    back = np.zeros((n, 3), dtype=np.int8)
    h, a, d = rows[0]
    for i in range(1, n):
        sh, sa, sd = rows[i]
        ch = (h, 0) if h >= a - switch_fee and h >= d - dead_fee else (
            (a - switch_fee, 1) if a - switch_fee >= d - dead_fee else (d - dead_fee, 2))
        ca = (a, 1) if a >= h - switch_fee and a >= d - dead_fee else (
            (h - switch_fee, 0) if h - switch_fee >= d - dead_fee else (d - dead_fee, 2))
        cd = (d, 2) if d >= h - dead_fee and d >= a - dead_fee else (
            (h - dead_fee, 0) if h >= a else (a - dead_fee, 1))
        back[i, 0], back[i, 1], back[i, 2] = ch[1], ca[1], cd[1]
        h, a, d = ch[0] + sh, ca[0] + sa, cd[0] + sd
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax([h, a, d]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(score: np.ndarray, segments, switch_fee: float, dead_fee: float) -> np.ndarray:
    """Decode each segment independently; frames outside every segment are -1."""
    out = np.full(len(score), -1, dtype=np.int8)
    for a, b in segments:
        out[a:b] = viterbi(score[a:b], switch_fee, dead_fee)
    return out


def ball_out(t: dict, off: np.ndarray, min_s: float = config.OUT_OF_PLAY_MIN_S) -> np.ndarray:
    """Law 9: off the pitch for min_s or longer means dead until the ball is next seen on it."""
    n = len(off)
    seen_on = ~np.isnan(t["bx"]) & ~off
    nxt = np.where(seen_on, np.arange(n), n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    nxt = np.concatenate((nxt, [n]))
    dead = np.zeros(n, dtype=bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], off.astype(np.int8), [0]))))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_s * FPS:
            dead[a:nxt[b]] = True
    return dead


def evidence(t: dict, tags: list, restarts: dict, off: np.ndarray, norm) -> dict:
    """Masks and restart frames the decoder can weigh."""
    n = len(t["bx"])
    frames, goal = [], np.zeros(n, dtype=bool)
    for tag in tags:
        kind = norm(tag["tags"].get("Type"))
        if norm(tag["code"]) == "set pieces" and kind in config.SET_PIECE_TYPES and id(tag) in restarts:
            frames.append(restarts[id(tag)][0])
        if kind == "kick off (after goal)":
            goal[max(0, tag["start_frame"] - int(GOAL_WINDOW_S * FPS)):tag["start_frame"]] = True
    return {"out": ball_out(t, off), "goal": goal, "off": off,
            "restarts": np.array(sorted(frames), dtype=np.int64)}


def _set_mask(n: int, restarts: np.ndarray, lead_s: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for r in restarts:
        mask[max(0, r - int(lead_s * FPS)):r] = True
    return mask


def _runs(codes: np.ndarray) -> list:
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(codes)]))
    return [(int(codes[a]), int(a), int(b)) for a, b in zip(starts, stops)]


def cleanup(codes: np.ndarray, segments, off: np.ndarray) -> np.ndarray:
    """Remove stoppages under MIN_DEAD_S (unless the ball was out for 1 s+) and plays under MIN_PLAY_S."""
    out = codes.copy()
    for s, e in segments:
        part, part_off = out[s:e], off[s:e]
        segs = _runs(part)
        for i in range(1, len(segs) - 1):
            label, a, b = segs[i]
            before, after = segs[i - 1][0], segs[i + 1][0]
            if label == D and before != D and after != D and b - a < MIN_DEAD_S * FPS:
                longest = 0
                if part_off[a:b].any():
                    longest = max(r[2] - r[1] for r in _runs(part_off[a:b].astype(np.int8)) if r[0] == 1)
                if before == after or longest < config.OUT_OF_PLAY_MIN_S * FPS:
                    part[a:b] = before
            elif label != D and before == D and after == D and b - a < MIN_PLAY_S * FPS:
                part[a:b] = D
    return out


def label(log_probs: np.ndarray, ev: dict, segments, settings: dict) -> np.ndarray:
    """Return state codes (0 H, 1 A, 2 D; -1 outside segments) for the given settings."""
    n = len(log_probs)
    score = log_probs.copy()
    sets = _set_mask(n, ev["restarts"], settings["set_lead_s"])
    forced = np.zeros(n, dtype=bool)
    for mask, weight in ((ev["out"], settings["w_out"]), (sets, settings["w_set"]), (ev["goal"], settings["w_goal"])):
        score[mask, D] += weight
        if weight >= HARD:
            forced |= mask
    codes = decode(score, segments, settings["switch_fee"], settings["dead_fee"])
    if settings["cleanup"]:
        codes = cleanup(codes, segments, ev["off"])
        codes[forced & (codes >= 0)] = D
    return codes
