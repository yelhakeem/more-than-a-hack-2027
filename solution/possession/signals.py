"""Low-level signals derived from the tracking: ball out of play and ball speed."""

import numpy as np

from possession import config


def off_pitch(t: dict) -> np.ndarray:
    """True where the ball is seen more than OFF_PITCH_MARGIN_M outside the pitch lines."""
    length, width = (float(v) for v in t["pitch"])
    m = config.OFF_PITCH_MARGIN_M
    return (t["bx"] < -m) | (t["bx"] > length + m) | (t["by"] < -m) | (t["by"] > width + m)


def ball_speed(t: dict) -> np.ndarray:
    """Ball speed in m/s from positions (centred difference over 2 * BALL_SPEED_SPAN frames)."""
    k = config.BALL_SPEED_SPAN
    speed = np.full(len(t["bx"]), np.nan)
    dx = t["bx"][2 * k:] - t["bx"][:-2 * k]
    dy = t["by"][2 * k:] - t["by"][:-2 * k]
    speed[k:-k] = np.hypot(dx, dy) * config.FPS / (2 * k)
    return speed
