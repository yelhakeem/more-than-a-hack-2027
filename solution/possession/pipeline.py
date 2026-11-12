"""End-to-end labelling: inputs -> features -> classifier -> decoder -> labels."""

import json
from dataclasses import dataclass

import joblib
import numpy as np

from possession import config, data, decode, features, model, restarts, signals
from possession.tags import norm


@dataclass
class Inputs:
    """Everything the classifier and decoder need for one match."""
    X: np.ndarray
    names: list
    segments: list
    evidence: dict


def prepare(t: dict, segments, tags: list, cap_s: float = config.TIME_CAP_S) -> Inputs:
    """Build features and decoder evidence for a match (provided or external)."""
    off = signals.off_pitch(t)
    found = restarts.find_restarts(t, signals.ball_speed(t), tags)
    X, names = features.build(t, segments, tags, found, off, cap_s)
    ev = decode.evidence(t, tags, found, off, norm)
    return Inputs(X=X, names=names, segments=list(segments), evidence=ev)


def load_model(model_path: str = config.MODEL_PATH, settings_path: str = config.SETTINGS_PATH) -> tuple:
    """Return the trained classifier and its settings."""
    with open(settings_path) as handle:
        settings = json.load(handle)
    return joblib.load(model_path), settings


def predict(inputs: Inputs, clf, settings: dict) -> np.ndarray:
    """State codes per frame (0 H, 1 A, 2 D; -1 outside the segments)."""
    if inputs.names != settings["features"]:
        raise ValueError("Feature list differs from the trained model; retrain with train.py.")
    rows = np.concatenate([np.arange(a, b) for a, b in inputs.segments])
    lp = np.zeros((len(inputs.X), 3))
    lp[rows] = model.log_probs(clf, inputs.X[rows])
    return decode.label(lp, inputs.evidence, inputs.segments, settings["settings"])


def label_match(t: dict, tags: list, clf, settings: dict) -> np.ndarray:
    """Labels ('H', 'A', 'D', 'X') for every video frame of the provided match."""
    inputs = prepare(t, data.play_segments(), tags, settings["cap_s"])
    codes = predict(inputs, clf, settings)
    labels = np.full(len(codes), "X", dtype="<U1")
    for _, first, last in data.graded_blocks():
        labels[first:last + 1] = "D"                      # graded frames before kick-off
    labels[codes >= 0] = np.array(config.STATES)[codes[codes >= 0]]
    return labels
