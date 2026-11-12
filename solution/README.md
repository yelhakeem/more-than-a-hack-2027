# Ball possession from tactical tracking: More Than A Hack 2027

Labels every frame of FC Barcelona (H) vs Athletic Club (A) as home possession (`H`), away
possession (`A`), dead ball (`D`) or not graded (`X`).

## Approach

```
tracking + Smart Tagging ──► features ──► classifier ──► decoder ──► frame labels
                                            ▲              ▲
                       trained and tuned on external data only (DFL open dataset)
```

- **Features** (`possession/features.py`, 109 per frame): ball position, speed and distance to each
  team; team shape, speed and movement; Smart Tagging set-piece windows; the restart located per
  the Laws of the Game (`possession/restarts.py`); time since the ball left the pitch; 1–10 s context.
- **Classifier** (`possession/model.py`): gradient-boosted trees giving P(H), P(A), P(D) per frame.
- **Decoder** (`possession/decode.py`): a Viterbi path with a cost for each change of state, so
  touches and duels do not fragment possessions. A ball more than 1 m off the pitch for 1 s or
  longer is always a dead ball (Law 9).

## Results on external data

Seven Bundesliga matches, each held out from training and degraded to the provided match's
tracking quality (ball missing on about 41% of frames). Decoder settings are re-tuned without
the scored match, so no match is scored with settings chosen on it.

| Metric | Score |
| --- | --- |
| spellF1 (ranking metric) | 0.553 |
| macroF1 | 0.897 |
| Per-frame accuracy | 91.3% |

`python solution/evaluate_external.py` reports how these scores change with ball visibility.
No model, setting or selection in the final pipeline was fitted to the provided match's ground
truth. The technical document discloses how the public 5-minute window was used earlier in
development.

## Setup

Python 3.11. From the repository root:

```
python -m pip install -r solution/requirements.txt
```

Place the match data in `data/` as described in `DATA.md`.

## Reproduce the submission

```
python solution/make_submission.py --out evaluation/results/<team_name>.csv
```

This loads the committed model (`solution/models/model.joblib` and `model.json`), writes the CSV
and validates its format with the organiser's own check. Runtime is a few minutes.

## Retrain from scratch (optional)

```
python solution/train.py
```

Downloads the DFL open dataset (about 2.4 GB, cached in `data/external/`), trains and tunes on it,
and rewrites `solution/models/`. Training is deterministic: the same files are produced.

## Layout

```
solution/
  make_submission.py     produce and validate the submission
  train.py               train the classifier and tune the decoder (external data)
  evaluate_external.py   robustness of the pipeline to ball visibility (external data)
  check_format.py        format check (organiser's validation function)
  possession/            the pipeline package
  models/                trained classifier and decoder settings
```

## Data and licences

- **Provided match**: tracking and Smart Tagging from the challenge organisers; not redistributed.
- **DFL open dataset**: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated
  dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195.
  https://doi.org/10.1038/s41597-025-04505-y. Data © Deutsche Fußball Liga, CC BY 4.0.
  Loaded with [floodlight](https://github.com/floodlight-sports/floodlight).
- Stoppage durations: Augste, C. & Cordes, O. (2016). *Game stoppages as a tactical means in
  soccer.* International Journal of Performance Analysis in Sport 16(3), 1053–1064.

## Development history

Earlier rule-based and hybrid versions (v1–v27) are preserved at the git tag `dev-history`.
