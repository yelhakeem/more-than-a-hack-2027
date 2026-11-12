"""External training and evaluation data: the DFL open dataset.

Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). An integrated dataset of spatiotemporal
and event data in elite soccer. Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y
Data (c) Deutsche Fussball Liga (DFL), licensed CC BY 4.0.

Seven Bundesliga matches with per-frame ball status and possession (25 fps). They are used to
train the classifier, tune the decoder and measure accuracy. To resemble the provided match,
each match is degraded to single-camera quality (degrade) and given stand-in Smart Tagging
built from its event data (standin_tags).
"""

import json
import os

import numpy as np

from possession import config

FPS = config.FPS
H, A, D = config.H, config.A, config.D
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")

# Degradation targets: label-free statistics of the provided match's tracking (DATA.md and
# the tracking summary): ball seen on 59.3% of graded frames; ball-missing stretches with
# median 10.7 s, 90th percentile 33.8 s, longest 221.5 s; ball seen off the pitch on 4.2% of
# the frames where it is seen; 20.5 of 22 players tracked on average.
BALL_MISSING_SHARE = 0.407
BALL_MISSING_SHARES = (0.35, 0.407, 0.45)
GAP_MEDIAN_S, GAP_P90_S, GAP_MAX_S, GAP_MIN_S = 10.7, 33.8, 221.5, 0.2
GAP_SIGMA = float(np.log(GAP_P90_S / GAP_MEDIAN_S) / 1.2816)
OFF_SEEN_SHARE = 0.042
EXIT_LEAD_S = 2.0
ANCHOR_SHARES = tuple(np.linspace(0, 1, 11))
PLAYER_MISSING_SHARE = 1 - 20.5 / 22
PLAYER_GAP_MEAN_S = 3.0
TEST_SEED_BASE = 1000

# Stand-in Smart Tagging, placed like the provided match's tags.
SET_WINDOW_S, SET_LEAD_S = 35.0, 7.6
KICKOFF_WINDOW_S = 5.0
TEAM_NAMES = {H: "Barcelona", A: "Athletic"}
EVENT_KINDS = (("throwin", "Throw-in"), ("cornerkick", "Corner Kick"), ("goalkick", "Goal Kick"),
               ("freekick", "Free Kick"), ("kickoff", "Kick Off"))


# --- Loading -----------------------------------------------------------------------------------

def _to_code(code_obj, mapping_words: dict, default: int) -> np.ndarray:
    """Map a floodlight Code to our integers using its definitions."""
    raw = np.asarray(code_obj.code)
    defs = getattr(code_obj, "definitions", None) or {}
    out = np.full(len(raw), default, dtype=np.int8)
    for value, meaning in defs.items():
        for word, ours in mapping_words.items():
            if word in str(meaning).lower():
                out[raw == value] = ours
    return out


def _convert(match_id: str) -> dict:
    """Download (if needed) and convert one match with floodlight."""
    from floodlight.io.datasets import IDSSEDataset

    dataset = IDSSEDataset(match_id=match_id)
    _, xy, possession, ballstatus, _, pitch = dataset.get(match_id, events=False)
    x0, x1 = (float(v) for v in pitch.xlim)
    y0, y1 = (float(v) for v in pitch.ylim)
    length, width = x1 - x0, y1 - y0
    parts = {k: [] for k in ("px", "py", "bx", "by", "label")}
    segments, team, start = [], None, 0
    for half in xy:
        groups = xy[half]
        names = {g.lower(): g for g in groups}
        home, away, ball = (groups[names[k]] for k in ("home", "away", "ball"))
        hx, ax = np.asarray(home.xy, dtype=np.float32), np.asarray(away.xy, dtype=np.float32)
        bxy = np.asarray(ball.xy, dtype=np.float32)
        n = len(bxy)
        players = np.concatenate([hx, ax], axis=1)
        parts["px"].append(players[:, 0::2] - x0)
        parts["py"].append(players[:, 1::2] - y0)
        parts["bx"].append(bxy[:, 0] - x0)
        parts["by"].append(bxy[:, 1] - y0)
        team_half = np.array([config.HOME] * (hx.shape[1] // 2) + [config.AWAY] * (ax.shape[1] // 2),
                             dtype=np.int8)
        if team is not None and len(team) != len(team_half):
            raise ValueError(f"{match_id}: player columns differ between halves")
        team = team_half
        alive = _to_code(ballstatus[half], {"alive": 1, "dead": 0}, default=-1)
        if (alive < 0).all():
            alive = (np.asarray(ballstatus[half].code) == 1).astype(np.int8)
        side = _to_code(possession[half], {"home": H, "away": A}, default=-1)
        if (side < 0).all():
            raw = np.asarray(possession[half].code)
            side = np.where(raw == 1, H, np.where(raw == 2, A, -1)).astype(np.int8)
        for i in range(1, n):
            if side[i] < 0:
                side[i] = side[i - 1]
        label = np.where(alive == 1, side, D).astype(np.int8)
        label[label < 0] = D
        parts["label"].append(label)
        segments.append((start, start + n))
        start += n
    out = {k: np.concatenate(v) for k, v in parts.items()}
    fps = getattr(xy[next(iter(xy))][names["ball"]], "framerate", FPS)
    if int(round(fps)) != FPS:
        raise ValueError(f"{match_id}: expected {FPS} fps, found {fps}")
    out.update(team=team, pitch=np.array([length, width], dtype=np.float32),
               segments=np.array(segments, dtype=np.int64))
    return out


def load_match(match_id: str) -> dict:
    """Tracking, labels and half boundaries of one match, cached under data/external/."""
    path = os.path.join(config.EXTERNAL_CACHE, f"{match_id}.npz")
    if os.path.isfile(path):
        return dict(np.load(path))
    data = _convert(match_id)
    os.makedirs(config.EXTERNAL_CACHE, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def load_events(match_id: str) -> list:
    """Restart events [(frame, kind, team)] of one match, cached as JSON."""
    path = os.path.join(config.EXTERNAL_CACHE, f"{match_id}_events.json")
    if os.path.isfile(path):
        with open(path) as handle:
            return [tuple(e) for e in json.load(handle)]
    from floodlight.io.datasets import IDSSEDataset

    segments = load_match(match_id)["segments"]
    events, *_ = IDSSEDataset(match_id=match_id).get(match_id, events=True, positions=False)
    out = []
    for index, half in enumerate(events):
        seg_start = int(segments[index][0])
        for group, obj in events[half].items():
            team = H if "home" in str(group).lower() else A if "away" in str(group).lower() else None
            df = obj.events
            if "gameclock" not in df.columns or "eID" not in df.columns:
                raise ValueError(f"{match_id}: event columns are {list(df.columns)}")
            for eid, clock in zip(df["eID"].astype(str), df["gameclock"].astype(float)):
                key = "".join(ch for ch in eid.lower() if ch.isalpha())
                for word, kind in EVENT_KINDS:
                    if word in key and team is not None and np.isfinite(clock):
                        out.append((seg_start + int(round(clock * FPS)), kind, int(team)))
                        break
    out.sort()
    os.makedirs(config.EXTERNAL_CACHE, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(out, handle)
    return out


# --- Degradation to single-camera quality ------------------------------------------------------

def _player_gaps(n: int, missing_share: float, gap_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Alternating seen / missing stretches with exponential lengths."""
    seen_mean = gap_mean * (1 - missing_share) / missing_share
    mask = np.zeros(n, dtype=bool)
    i, missing = 0, bool(rng.random() < missing_share)
    while i < n:
        length = max(1, int(rng.exponential((gap_mean if missing else seen_mean) * FPS)))
        if missing:
            mask[i:i + length] = True
        i += length
        missing = not missing
    return mask


def _ball_off_lines(t: dict) -> np.ndarray:
    length, width = (float(v) for v in t["pitch"])
    x, y = t["bx"], t["by"]
    return (x < 0) | (x > length) | (y < 0) | (y > width)


def _ball_gaps(n: int, share: float, exits: np.ndarray, anchor: float, rng: np.random.Generator,
               gap_scale: float = 1.0) -> np.ndarray:
    """Ball-missing mask: lognormal gap lengths, a share of them starting at ball exits."""
    mask = np.zeros(n, dtype=bool)
    mu = np.log(GAP_MEDIAN_S * gap_scale * FPS)
    lo, hi = int(GAP_MIN_S * FPS), int(GAP_MAX_S * gap_scale * FPS)
    lead = int(EXIT_LEAD_S * FPS)
    while mask.mean() < share:
        length = int(np.clip(rng.lognormal(mu, GAP_SIGMA), lo, hi))
        if exits.size and rng.random() < anchor:
            start = int(rng.choice(exits)) - int(rng.integers(0, lead + 1))
        else:
            start = int(rng.integers(0, n))
        mask[max(0, start):start + length] = True
    return mask


def degrade(t: dict, rng: np.random.Generator, missing_share: float = None,
            gap_scale: float = 1.0) -> dict:
    """Return a copy of a match with ball and player gaps like the provided match's tracking.

    Args:
        t: clean match from load_match.
        rng: random generator (determines the gaps).
        missing_share: share of frames without the ball; drawn from BALL_MISSING_SHARES if None.
        gap_scale: multiplies the typical gap length (1.0 = the provided match).
    """
    out = dict(t)
    n = len(t["bx"])
    share = missing_share if missing_share is not None else float(rng.choice(BALL_MISSING_SHARES))
    off = _ball_off_lines(t)
    exits = np.flatnonzero(off[1:] & ~off[:-1]) + 1
    best = None
    for anchor in ANCHOR_SHARES:
        trial_rng = np.random.default_rng(rng.integers(2 ** 32))
        lost = _ball_gaps(n, share, exits, anchor, trial_rng, gap_scale)
        seen = ~lost
        off_seen = float(off[seen].mean()) if seen.any() else 0.0
        if best is None or abs(off_seen - OFF_SEEN_SHARE) < best[0]:
            best = (abs(off_seen - OFF_SEEN_SHARE), lost, anchor, off_seen)
        if off_seen <= OFF_SEEN_SHARE:
            break
    _, lost, anchor, off_seen = best
    out["degrade_info"] = np.array([share, anchor, off_seen, lost.mean()])
    out["bx"], out["by"] = t["bx"].copy(), t["by"].copy()
    out["bx"][lost] = np.nan
    out["by"][lost] = np.nan
    px, py = t["px"].copy(), t["py"].copy()
    for j in range(px.shape[1]):
        drop = _player_gaps(n, PLAYER_MISSING_SHARE, PLAYER_GAP_MEAN_S, rng)
        px[drop, j] = np.nan
        py[drop, j] = np.nan
    out["px"], out["py"] = px, py
    return out


def swap_teams(t: dict) -> dict:
    """The same match seen from the other side: home and away (and H and A) exchanged."""
    out = dict(t)
    out["team"] = (1 - t["team"]).astype(np.int8)
    if "label" in t:
        lab = t["label"].copy()
        lab[t["label"] == H], lab[t["label"] == A] = A, H
        out["label"] = lab
    return out


def test_copy(match_id: str, missing_share: float = BALL_MISSING_SHARE, gap_scale: float = 1.0) -> dict:
    """The fixed held-out evaluation copy of a match."""
    seed = TEST_SEED_BASE + MATCHES.index(match_id)
    return degrade(load_match(match_id), np.random.default_rng(seed), missing_share, gap_scale)


# --- Stand-in Smart Tagging --------------------------------------------------------------------

def standin_tags(match_id: str, segments, swapped: bool = False) -> list:
    """Smart-Tagging-like set-piece windows built from a match's restart events."""
    tags = []
    starts = {int(a) for a, _ in segments}
    for frame, kind, team in load_events(match_id):
        if kind == "Kick Off":
            near_start = any(abs(frame - s) < 5 * FPS for s in starts)
            kind = "Kick Off (Start)" if near_start else "Kick Off (After Goal)"
            start = frame - int(round(0.68 * KICKOFF_WINDOW_S * FPS))
            end = start + int(KICKOFF_WINDOW_S * FPS)
        else:
            start = frame - int(SET_LEAD_S * FPS)
            end = start + int(SET_WINDOW_S * FPS)
        if kind in ("Throw-in", "Kick Off (Start)"):
            side = "N/A"
        else:
            side = TEAM_NAMES[(1 - team) if swapped else team]
        tags.append({"code": "Set Pieces", "start_frame": max(0, start), "end_frame": end,
                     "start_s": start / FPS, "end_s": end / FPS, "tags": {"Type": kind, "Team": side}})
    return tags
