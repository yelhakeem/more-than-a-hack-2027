"""Load the tracking data once, save a fast copy, and print a summary.

What this file does, in plain words:

  1. Reads the big tracking text file (one line per video frame) and turns it into tables of
     numbers: where every tracked player is, which team they belong to, and where the ball is.
  2. Saves those tables to data/cache/tracking.npz. Reading the text file takes a minute or two;
     reading the saved copy takes about a second, so every later script starts fast.
     The cache lives inside data/, which git ignores, so it never ends up in the public repo.
  3. Prints only summary numbers (percentages and averages), never raw positions.

Run it from the project folder:

    python solution/tracking.py

Other scripts use it like this:

    from tracking import load_tracking
    t = load_tracking()
    t["ball_x"][frame]      # ball position in metres along the pitch, NaN if not seen
"""

import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

CACHE_PATH = os.path.join(DATA_DIR, "cache", "tracking.npz")
CACHE_VERSION = 1

HOME, AWAY = 0, 1


def track_teams(meta_path):
    """For each track slot, 0 if it belongs to the home team (Barcelona), 1 if away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    home_id = team_ids[0]  # the first team listed is the home side (FIFA/EPTS convention)
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def _xy(group):
    """'x,y,speed' -> (x, y, speed) as floats. 'NaN' becomes nan automatically."""
    parts = group.split(",")
    speed = float(parts[2]) if len(parts) > 2 and parts[2] else np.nan
    return float(parts[0]), float(parts[1]), speed


def parse_raw(raw_path, n_slots, n_frames, pitch):
    """Read the raw tracking file into arrays, one row per VIDEO frame.

    Each line looks like  <frame_id>:<player>;<player>;...:<ball>  and each group is x,y,speed.
    frame_id counts from 1, video frames from 0, so row = frame_id - 1.
    Positions arrive as 0-1 fractions of the pitch; we convert them to metres so that distances
    are real distances (the pitch is longer than it is wide).
    """
    length, width = pitch
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full(n_frames, np.nan, dtype=np.float32)
    bs = np.full(n_frames, np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            if not 0 <= row < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                x, y, s = _xy(group)
                px[row, slot], py[row, slot], ps[row, slot] = x * length, y * width, s
            if parts[2] and parts[2][0] != "N":
                x, y, s = _xy(parts[2])
                bx[row], by[row], bs[row] = x * length, y * width, s
    return px, py, ps, bx, by, bs


def load_tracking(rebuild=False):
    """The tracking as a dict of numpy arrays, built from the raw file the first time only.

    Keys: px, py, ps   players' x, y (metres) and speed (m/s); shape (frames, slots)
          bx, by, bs   ball x, y (metres) and speed; shape (frames,)
          team         team of each slot: 0 home, 1 away
          pitch        (length, width) in metres
    Missing observations are NaN.
    """
    if not rebuild and os.path.isfile(CACHE_PATH):
        cached = dict(np.load(CACHE_PATH))
        if int(cached.get("version", -1)) == CACHE_VERSION:
            return cached

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = track_teams(meta_path)
    pitch = GAMES[GAME]["pitch"]
    print("Reading {} (takes a minute or two, only the first time) ...".format(
        os.path.basename(raw_path)))
    start = time.time()
    px, py, ps, bx, by, bs = parse_raw(raw_path, len(team), source_frames(), pitch)
    print("  done in {:.0f} s".format(time.time() - start))

    data = dict(px=px, py=py, ps=ps, bx=bx, by=by, bs=bs, team=team,
                pitch=np.array(pitch, dtype=np.float32), version=np.array(CACHE_VERSION))
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    np.savez(CACHE_PATH, **data)
    print("  saved fast copy to {}".format(os.path.relpath(CACHE_PATH, ROOT)))
    return data


def graded_mask(n_frames):
    """True on the frames that are scored (the two halves), False elsewhere."""
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def summary(t):
    """Print summary numbers only -- safe to share. Never prints positions."""
    n_frames = len(t["bx"])
    graded = graded_mask(n_frames)
    length, width = [float(v) for v in t["pitch"]]
    ball_seen = ~np.isnan(t["bx"])
    seen_players = ~np.isnan(t["px"])
    home_slots = t["team"] == HOME

    print("\nSUMMARY (graded frames only, {:,} frames)".format(int(graded.sum())))
    print("  track slots in metadata:        {} ({} home, {} away)".format(
        len(t["team"]), int(home_slots.sum()), int((~home_slots).sum())))
    print("  ball seen:                      {:.1f}%  (DATA.md says 59.3%)".format(
        100 * ball_seen[graded].mean()))
    per_frame = seen_players[graded].sum(axis=1)
    print("  players tracked per frame:      {:.1f} on average  (DATA.md says 20.5)".format(
        per_frame.mean()))
    print("    home / away:                  {:.1f} / {:.1f}".format(
        seen_players[graded][:, home_slots].sum(axis=1).mean(),
        seen_players[graded][:, ~home_slots].sum(axis=1).mean()))
    print("  frames with fewer than 15:      {:.1f}%".format(100 * (per_frame < 15).mean()))

    seen = graded & ball_seen
    off = seen & ((t["bx"] < 0) | (t["bx"] > length) | (t["by"] < 0) | (t["by"] > width))
    print("  ball seen OFF the pitch:        {:.1f}% of frames where it is seen".format(
        100 * off.sum() / max(seen.sum(), 1)))

    # How long does the ball stay missing? Long gaps are where guessing gets hard.
    gaps, run = [], 0
    for missing in ~ball_seen[graded]:
        if missing:
            run += 1
        elif run:
            gaps.append(run)
            run = 0
    if run:
        gaps.append(run)
    gaps = np.array(gaps) if gaps else np.array([0])
    print("  ball-missing stretches:         {:,}".format(len(gaps)))
    print("    median / 90th pct / longest:  {:.1f} s / {:.1f} s / {:.1f} s".format(
        np.median(gaps) / 25, np.percentile(gaps, 90) / 25, gaps.max() / 25))
    print("    share over 2 seconds:         {:.1f}%".format(100 * (gaps > 50).mean()))


if __name__ == "__main__":
    summary(load_tracking(rebuild="--rebuild" in sys.argv))
