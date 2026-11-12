"""Constants and paths shared across the pipeline.

Every threshold here comes from the Laws of the Game, published match statistics or the
challenge documentation; none was fitted to the provided match.
"""

import os

from games import DATA_DIR
from possession import REPO_ROOT

# Units and label encoding
FPS = 25
HOME, AWAY = 0, 1
H, A, D = 0, 1, 2
STATES = ("H", "A", "D")

# Provided match (DATA.md / FAQ.md)
KICKOFF_FRAMES = {1: 11185, 2: 113742}

# Ball out of play (Law 9); a 1 m margin absorbs tracking noise on the lines.
OFF_PITCH_MARGIN_M = 1.0
OUT_OF_PLAY_MIN_S = 1.0
BALL_SPEED_SPAN = 2                    # frames either side for the centred speed estimate

# Smart Tagging set pieces
SET_PIECE_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)")
BEFORE_TAG_S, AFTER_TAG_S = 4.0, 3.0   # search window around each tag
RESTART_FRACTION = 0.68                # restart position inside a kick-off tag (both kick-offs)

# Features
TIME_CAP_S = 60.0                      # cap for "seconds since / until" features

# Paths
SOLUTION_DIR = os.path.join(REPO_ROOT, "solution")
MODEL_DIR = os.path.join(SOLUTION_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
SETTINGS_PATH = os.path.join(MODEL_DIR, "model.json")
EXTERNAL_CACHE = os.path.join(REPO_ROOT, "data", "external")
TRACKING_CACHE = os.path.join(DATA_DIR, "cache", "tracking.npz")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "results", "submission.csv")
