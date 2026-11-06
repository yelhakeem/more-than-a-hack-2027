"""Load the tactical tracking into numpy arrays, cached for fast reuse.

Usage:
    python solution/tracking.py            # build the cache if needed, print a summary
    python solution/tracking.py --rebuild  # force a re-parse of the raw file

The cache lives in data/cache/, which git ignores. The summary prints aggregates only.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

CACHE_PATH = os.path.join(DATA_DIR, "cache", "tracking.npz")
CACHE_VERSION = 1
FPS = 25
HOME, AWAY = 0, 1


def track_teams(meta_path: str) -> np.ndarray:
    """Team of each track slot (HOME or AWAY). The first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams in {meta_path}, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def _parse_group(group: str) -> tuple:
    """'x,y,speed' -> floats; 'NaN' parses to nan."""
    parts = group.split(",")
    speed = float(parts[2]) if len(parts) > 2 and parts[2] else np.nan
    return float(parts[0]), float(parts[1]), speed


def parse_raw(raw_path: str, n_slots: int, n_frames: int, pitch: tuple) -> dict:
    """Parse '<frame_id>:<player>;...:<ball>' lines into arrays indexed by video frame.

    Positions are converted from 0-1 pitch fractions to metres; video frame = frame_id - 1.
    """
    length, width = pitch
    shape = (n_frames, n_slots)
    px, py, ps = (np.full(shape, np.nan, dtype=np.float32) for _ in range(3))
    bx, by, bs = (np.full(n_frames, np.nan, dtype=np.float32) for _ in range(3))

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if slot < n_slots and group and group[0] != "N":
                    x, y, s = _parse_group(group)
                    px[row, slot], py[row, slot], ps[row, slot] = x * length, y * width, s
            if parts[2] and parts[2][0] != "N":
                x, y, s = _parse_group(parts[2])
                bx[row], by[row], bs[row] = x * length, y * width, s

    return dict(px=px, py=py, ps=ps, bx=bx, by=by, bs=bs)


def load_tracking(rebuild: bool = False) -> dict:
    """Tracking arrays, from the cache when available.

    Keys: px, py, ps (frames x slots: metres, metres, m/s); bx, by, bs (frames);
    team (slots: HOME/AWAY); pitch (length, width in metres). Missing values are NaN.
    """
    if not rebuild and os.path.isfile(CACHE_PATH):
        cached = dict(np.load(CACHE_PATH))
        if int(cached.get("version", -1)) == CACHE_VERSION:
            return cached

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = track_teams(meta_path)
    pitch = GAMES[GAME]["pitch"]

    print(f"Parsing {os.path.basename(raw_path)} (first run only) ...")
    start = time.time()
    data = parse_raw(raw_path, len(team), source_frames(), pitch)
    print(f"  done in {time.time() - start:.0f} s")

    data.update(team=team, pitch=np.array(pitch, dtype=np.float32),
                version=np.array(CACHE_VERSION))
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, **data)
    print(f"  cached to {os.path.relpath(CACHE_PATH, ROOT)}")
    return data


def graded_mask(n_frames: int) -> np.ndarray:
    """True on frames inside the two graded halves."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def run_lengths(flags: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs."""
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def summary(t: dict) -> None:
    """Print aggregate statistics over graded frames. No positions are printed."""
    graded = graded_mask(len(t["bx"]))
    length, width = (float(v) for v in t["pitch"])
    ball_seen = ~np.isnan(t["bx"])
    players_seen = ~np.isnan(t["px"])
    home = t["team"] == HOME

    per_frame = players_seen[graded].sum(axis=1)
    seen = graded & ball_seen
    off = seen & ((t["bx"] < 0) | (t["bx"] > length) | (t["by"] < 0) | (t["by"] > width))
    gaps = run_lengths(~ball_seen[graded])
    if gaps.size == 0:
        gaps = np.array([0])

    print(f"\nSUMMARY (graded frames only, {int(graded.sum()):,} frames)")
    print(f"  track slots in metadata:        {len(home)} ({int(home.sum())} home, "
          f"{int((~home).sum())} away)")
    print(f"  ball seen:                      {100 * ball_seen[graded].mean():.1f}%  "
          f"(DATA.md says 59.3%)")
    print(f"  players tracked per frame:      {per_frame.mean():.1f} on average  "
          f"(DATA.md says 20.5)")
    print(f"    home / away:                  {players_seen[graded][:, home].sum(axis=1).mean():.1f}"
          f" / {players_seen[graded][:, ~home].sum(axis=1).mean():.1f}")
    print(f"  frames with fewer than 15:      {100 * (per_frame < 15).mean():.1f}%")
    print(f"  ball seen OFF the pitch:        {100 * off.sum() / max(seen.sum(), 1):.1f}% "
          f"of frames where it is seen")
    print(f"  ball-missing stretches:         {len(gaps):,}")
    print(f"    median / 90th pct / longest:  {np.median(gaps) / FPS:.1f} s / "
          f"{np.percentile(gaps, 90) / FPS:.1f} s / {gaps.max() / FPS:.1f} s")
    print(f"    share over 2 seconds:         {100 * (gaps > 2 * FPS).mean():.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="re-parse the raw tracking file")
    summary(load_tracking(rebuild=parser.parse_args().rebuild))


if __name__ == "__main__":
    main()
