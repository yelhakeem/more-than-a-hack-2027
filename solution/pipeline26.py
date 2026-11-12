"""v26's decision step, shared by tuning (external data) and labelling (our match).

  score = model log-probabilities
        + w_out  on "dead" where Law 9 says the ball is out (v24.ball_out)
        + w_set  on "dead" for set_lead_s seconds before each set-piece restart (restarts22)
        + w_goal on "dead" for the 41 s before each kick-off after a goal
  decode with the switch / dead fees, then (optionally) the cleanup of v5 + v20.
A weight of HARD forces the state, which reproduces v25's hard overrides.
"""

import numpy as np

import model18

FPS = 25
H, A, D = 0, 1, 2
HARD = 1000.0
GOAL_WINDOW_S = 41.0
MIN_DEAD_S, MIN_PLAY_S, OUT_EVIDENCE_S = 3.0, 1.0, 1.0
DEFAULTS = dict(switch_fee=8, dead_fee=48, w_out=HARD, w_set=HARD, set_lead_s=12.0,
                w_goal=HARD, cleanup=1)


def ball_out(t: dict, off: np.ndarray, min_s: float = OUT_EVIDENCE_S) -> np.ndarray:
    """Law 9: off the pitch for min_s+, dead until the ball is next seen on the pitch."""
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


def masks(t: dict, tags: list, found: dict, off: np.ndarray, norm, set_types) -> dict:
    """Restart frames and the Law 9 / goal masks (the set-piece mask depends on set_lead_s)."""
    n = len(t["bx"])
    restarts, goal = [], np.zeros(n, dtype=bool)
    for tag in tags:
        kind = norm(tag["tags"].get("Type"))
        if norm(tag["code"]) == "set pieces" and kind in set_types and id(tag) in found:
            restarts.append(found[id(tag)][0])
        if kind == "kick off (after goal)":
            goal[max(0, tag["start_frame"] - int(GOAL_WINDOW_S * FPS)):tag["start_frame"]] = True
    return {"out": ball_out(t, off), "goal": goal, "restarts": np.array(sorted(restarts), dtype=np.int64),
            "off": off}


def set_mask(n: int, restarts: np.ndarray, lead_s: float) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for r in restarts:
        mask[max(0, r - int(lead_s * FPS)):r] = True
    return mask


def runs(codes: np.ndarray) -> list:
    change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(codes)]))
    return [(int(codes[a]), int(a), int(b)) for a, b in zip(starts, stops)]


def cleanup(codes: np.ndarray, segments, off: np.ndarray) -> np.ndarray:
    """v5 + v20: stoppages under 3 s go to the team that had the ball (unless the ball was off
    the pitch for 1 s+ inside them); play under 1 s between two stoppages becomes dead."""
    out = codes.copy()
    for s, e in segments:
        part, part_off = out[s:e], off[s:e]
        segs = runs(part)
        for i in range(1, len(segs) - 1):
            label, a, b = segs[i]
            before, after = segs[i - 1][0], segs[i + 1][0]
            if label == D and before != D and after != D and b - a < MIN_DEAD_S * FPS:
                longest = 0
                if part_off[a:b].any():
                    longest = max(r[2] - r[1] for r in runs(part_off[a:b].astype(np.int8)) if r[0] == 1)
                if before == after or longest < OUT_EVIDENCE_S * FPS:
                    part[a:b] = before
            elif label != D and before == D and after == D and b - a < MIN_PLAY_S * FPS:
                part[a:b] = D
    return out


def label(log_probs: np.ndarray, m: dict, segments, p: dict) -> np.ndarray:
    n = len(log_probs)
    score = log_probs.copy()
    sets = set_mask(n, m["restarts"], p["set_lead_s"])
    forced = np.zeros(n, dtype=bool)
    for mask, w in ((m["out"], p["w_out"]), (sets, p["w_set"]), (m["goal"], p["w_goal"])):
        score[mask, D] += w
        if w >= HARD:
            forced |= mask
    codes = model18.decode(score, segments, p["switch_fee"], p["dead_fee"])
    if p["cleanup"]:
        codes = cleanup(codes, segments, m["off"])
        codes[forced & (codes >= 0)] = D                 # hard evidence has the last word
    return codes
