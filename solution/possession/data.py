"""Loading the provided match (tracking, Smart Tagging) and writing a submission."""

import logging
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from games import GAME, GAMES, half_video_blocks, require, source_frames
from load_smart_tagging import load as _load_tags
from possession import config

LOG = logging.getLogger(__name__)
_CACHE_VERSION = 1


def _track_teams(meta_path: str) -> np.ndarray:
    """Team (HOME / AWAY) of each track slot; the first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError(f"expected 2 teams in {meta_path}, found {len(team_ids)}")
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([config.HOME if t == team_ids[0] else config.AWAY for t in slots], dtype=np.int8)


def _parse_group(group: str) -> tuple:
    """Parse an 'x,y,speed' group; a missing speed becomes NaN."""
    parts = group.split(",")
    speed = float(parts[2]) if len(parts) > 2 and parts[2] else np.nan
    return float(parts[0]), float(parts[1]), speed


def _parse_raw(raw_path: str, n_slots: int, n_frames: int, pitch: tuple) -> dict:
    """Parse '<frame_id>:<players>:<ball>' lines into arrays indexed by video frame (id - 1)."""
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
    """Return the match tracking in metres, cached under data/cache/.

    Keys: px, py, ps (frames x slots); bx, by, bs (frames); team (slots); pitch (length, width).
    Missing values are NaN.
    """
    if not rebuild and os.path.isfile(config.TRACKING_CACHE):
        cached = dict(np.load(config.TRACKING_CACHE))
        if int(cached.get("version", -1)) == _CACHE_VERSION:
            return cached
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = _track_teams(meta_path)
    pitch = GAMES[GAME]["pitch"]
    LOG.info("Parsing %s (first run only)", os.path.basename(raw_path))
    start = time.time()
    data = _parse_raw(raw_path, len(team), source_frames(), pitch)
    data.update(team=team, pitch=np.array(pitch, dtype=np.float32), version=np.array(_CACHE_VERSION))
    os.makedirs(os.path.dirname(config.TRACKING_CACHE), exist_ok=True)
    np.savez(config.TRACKING_CACHE, **data)
    LOG.info("Parsed and cached in %.0f s", time.time() - start)
    return data


def load_tags() -> list:
    """Smart Tagging events of the match, via the organiser's starter loader."""
    return _load_tags()


def graded_blocks() -> list:
    """[(half, first_frame, last_frame)] of the graded halves (inclusive)."""
    return half_video_blocks(GAME)


def play_segments() -> list:
    """[(start, stop)] of each half from its kick-off to its last graded frame."""
    return [(max(first, config.KICKOFF_FRAMES[half]), last + 1) for half, first, last in graded_blocks()]


def write_submission(path: str, labels: np.ndarray) -> None:
    """Write the 'frame,label' CSV expected by the scorer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{i},{label}\n" for i, label in enumerate(labels))
