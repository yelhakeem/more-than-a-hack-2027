"""External training data: the open DFL Bundesliga matches, in the same format as our tracking.

Dataset: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "An integrated dataset of
spatiotemporal and event data in elite soccer." Scientific Data 12, 195.
https://doi.org/10.1038/s41597-025-04505-y  Data (c) Deutsche Fussball Liga (DFL), CC BY 4.0.

These are OTHER games, so training and tuning on them is allowed (RULES.md, and the organisers'
clarification: "If you need to train (or retrain), you MUST use only external data").

Each match gives, per frame at 25 fps: every player's and the ball's position, which team has
the ball, and whether the ball is alive or dead. That is the same kind of information the
challenge's ground truth is built from, so it can teach a model what "possession" and "dead
ball" look like in tracking data.

Usage:
    python -m pip install floodlight
    python solution/dfl.py            # download + convert all 7 matches (~2.4 GB download, once)

The converted matches are cached in data/external/ (git ignores data/).
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from tracking import AWAY, FPS, HOME  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "external")
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
H, A, D = 0, 1, 2

# Degrading the clean stadium tracking so it looks like tracking from a single tactical video.
# Targets are label-free statistics of our match's tracking (python solution/tracking.py;
# the first two are also published in DATA.md):
#   ball seen 59.3% of graded frames         -> 40.7% missing
#   ball-missing stretches: median 10.7 s, 90th percentile 33.8 s, longest 221.5 s
#   ball seen off the pitch on 4.2% of the frames where it is seen
#   players tracked: 20.5 of 22 on average
BALL_MISSING_SHARE = 0.407
BALL_MISSING_SHARES = (0.35, 0.407, 0.45)     # training varies around it a little
GAP_MEDIAN_S, GAP_P90_S, GAP_MAX_S, GAP_MIN_S = 10.7, 33.8, 221.5, 0.2
GAP_SIGMA = float(np.log(GAP_P90_S / GAP_MEDIAN_S) / 1.2816)   # lognormal through median and p90
OFF_SEEN_SHARE = 0.042
EXIT_LEAD_S = 2.0                              # a gap anchored to a ball exit starts up to 2 s before it
ANCHOR_SHARES = tuple(np.linspace(0, 1, 11))   # tried in turn to match OFF_SEEN_SHARE
PLAYER_MISSING_SHARE = 1 - 20.5 / 22           # ~7% of player-frames
PLAYER_GAP_MEAN_S = 3.0


def _cache_path(match_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{match_id}.npz")


def _to_code(code_obj, mapping_words: dict, default: int) -> np.ndarray:
    """Floodlight Code -> our integers, using its definitions when present."""
    raw = np.asarray(code_obj.code)
    defs = getattr(code_obj, "definitions", None) or {}
    out = np.full(len(raw), default, dtype=np.int8)
    for value, meaning in defs.items():
        for word, ours in mapping_words.items():
            if word in str(meaning).lower():
                out[raw == value] = ours
    return out


def convert(match_id: str) -> dict:
    """Download (if needed) and convert one match with floodlight."""
    from floodlight.io.datasets import IDSSEDataset   # only needed here

    dataset = IDSSEDataset(match_id=match_id)
    _, xy, possession, ballstatus, _, pitch = dataset.get(match_id, events=False)
    x0, x1 = (float(v) for v in pitch.xlim)
    y0, y1 = (float(v) for v in pitch.ylim)
    length, width = x1 - x0, y1 - y0

    parts = {k: [] for k in ("px", "py", "bx", "by", "label")}
    segments, team, start = [], None, 0
    for half in xy:                                   # firstHalf, secondHalf (in order)
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
        team_half = np.array([HOME] * (hx.shape[1] // 2) + [AWAY] * (ax.shape[1] // 2), dtype=np.int8)
        if team is not None and len(team) != len(team_half):
            raise ValueError(f"{match_id}: player columns differ between halves")
        team = team_half

        alive = _to_code(ballstatus[half], {"alive": 1, "dead": 0}, default=-1)
        if (alive < 0).all():                         # no readable definitions: DFL uses 1 = alive
            alive = (np.asarray(ballstatus[half].code) == 1).astype(np.int8)
        side = _to_code(possession[half], {"home": H, "away": A}, default=-1)
        if (side < 0).all():                          # DFL uses 1 = home, 2 = away
            raw = np.asarray(possession[half].code)
            side = np.where(raw == 1, H, np.where(raw == 2, A, -1)).astype(np.int8)
        for i in range(1, n):                         # rare gaps: keep the last known team
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
    path = _cache_path(match_id)
    if os.path.isfile(path):
        return dict(np.load(path))
    data = convert(match_id)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


def _gaps(n: int, missing_share: float, gap_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Player-track gaps: alternating seen / missing stretches of random length."""
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


def ball_off(t: dict) -> np.ndarray:
    """Ball outside the pitch lines (same test as tracking.py's summary)."""
    length, width = (float(v) for v in t["pitch"])
    x, y = t["bx"], t["by"]
    return (x < 0) | (x > length) | (y < 0) | (y > width)


def _ball_gaps(n: int, share: float, exits: np.ndarray, anchor: float,
               rng: np.random.Generator) -> np.ndarray:
    """Missing-ball mask: lognormal gap lengths like our video's, some starting at ball exits."""
    mask = np.zeros(n, dtype=bool)
    mu = np.log(GAP_MEDIAN_S * FPS)
    lo, hi = int(GAP_MIN_S * FPS), int(GAP_MAX_S * FPS)
    lead = int(EXIT_LEAD_S * FPS)
    while mask.mean() < share:
        length = int(np.clip(rng.lognormal(mu, GAP_SIGMA), lo, hi))
        if exits.size and rng.random() < anchor:
            start = int(rng.choice(exits)) - int(rng.integers(0, lead + 1))
        else:
            start = int(rng.integers(0, n))
        mask[max(0, start):start + length] = True
    return mask


def degrade(t: dict, rng: np.random.Generator, missing_share: float = None) -> dict:
    """Copy of a clean match with ball and player gaps like our tactical-video tracking.

    The share of gaps that start where the ball leaves the pitch is chosen so the ball is seen
    off the pitch about as often as in our match (OFF_SEEN_SHARE): in a single video the
    camera usually loses a ball that goes out.
    """
    out = dict(t)
    n = len(t["bx"])
    share = missing_share if missing_share is not None else float(rng.choice(BALL_MISSING_SHARES))
    off = ball_off(t)
    exits = np.flatnonzero(off[1:] & ~off[:-1]) + 1
    best = None
    for anchor in ANCHOR_SHARES:
        trial_rng = np.random.default_rng(rng.integers(2 ** 32))
        lost = _ball_gaps(n, share, exits, anchor, trial_rng)
        seen = ~lost
        off_seen = float(off[seen].mean()) if seen.any() else 0.0
        if best is None or abs(off_seen - OFF_SEEN_SHARE) < best[0]:
            best = (abs(off_seen - OFF_SEEN_SHARE), lost, anchor, off_seen)
        if off_seen <= OFF_SEEN_SHARE:
            break                                   # more anchoring only hides more off-pitch balls
    _, lost, anchor, off_seen = best
    out["degrade_info"] = np.array([share, anchor, off_seen, lost.mean()])
    out["bx"], out["by"] = t["bx"].copy(), t["by"].copy()
    out["bx"][lost] = np.nan
    out["by"][lost] = np.nan
    px, py = t["px"].copy(), t["py"].copy()
    for j in range(px.shape[1]):
        drop = _gaps(n, PLAYER_MISSING_SHARE, PLAYER_GAP_MEAN_S, rng)
        px[drop, j] = np.nan
        py[drop, j] = np.nan
    out["px"], out["py"] = px, py
    return out


def swap_teams(t: dict) -> dict:
    """The same match seen from the other side: home <-> away, H <-> A."""
    out = dict(t)
    out["team"] = (1 - t["team"]).astype(np.int8)
    if "label" in t:
        lab = t["label"].copy()
        lab[t["label"] == H], lab[t["label"] == A] = A, H
        out["label"] = lab
    return out


def main() -> None:
    print("DFL OPEN DATA (external, CC BY 4.0)")
    for match_id in MATCHES:
        try:
            t = load_match(match_id)
        except Exception as err:
            print(f"  {match_id}: failed: {type(err).__name__}: {err}")
            continue
        lab = t["label"]
        dead = lab == D
        runs = 1 + np.count_nonzero(lab[1:] != lab[:-1])
        print(f"  {match_id}: {len(lab):,} frames, {int((t['team'] == HOME).sum())}+"
              f"{int((t['team'] == AWAY).sum())} player columns, pitch {t['pitch'][0]:.1f} x "
              f"{t['pitch'][1]:.1f} m, dead {100 * dead.mean():.1f}%, "
              f"H {100 * np.mean(lab == H):.1f}%, A {100 * np.mean(lab == A):.1f}%, "
              f"label runs {runs}, ball missing {100 * np.isnan(t['bx']).mean():.1f}%")


if __name__ == "__main__":
    main()
