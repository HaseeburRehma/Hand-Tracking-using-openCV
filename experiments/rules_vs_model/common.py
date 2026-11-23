"""Shared definitions for the rules-versus-model experiment.

One CSV row = one camera frame in which a hand was detected.
Landmark coordinates are MediaPipe's normalised values (x, y in 0..1 of the
image width/height, z relative to the wrist), exactly as the detector returns
them. Nothing is derived or cleaned at recording time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

# The five commands of the original project (CarTracking.py).
CLASSES = ["thumb", "index", "middle", "pinky", "none"]

INSTRUCTIONS = {
    "thumb": "THUMB up, other fingers closed  (move straight)",
    "index": "INDEX finger up, others closed  (turn right)",
    "middle": "MIDDLE finger up, others closed  (turn left)",
    "pinky": "LITTLE finger up, others closed  (move back)",
    "none": "CLOSED FIST, no finger raised  (stop)",
}

N_LANDMARKS = 21
META_COLUMNS = [
    "person", "session", "condition", "label", "frame", "t_ms",
    "img_w", "img_h", "handedness",
]
X_COLUMNS = [f"x{i}" for i in range(N_LANDMARKS)]
Y_COLUMNS = [f"y{i}" for i in range(N_LANDMARKS)]
Z_COLUMNS = [f"z{i}" for i in range(N_LANDMARKS)]
COLUMNS = META_COLUMNS + X_COLUMNS + Y_COLUMNS + Z_COLUMNS


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read every recording CSV in data_dir into one table."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No recordings found in {data_dir}. Run record.py first.")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"person": str, "session": str, "condition": str})
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"{f.name} is missing columns: {missing[:5]}")
        bad = sorted(set(df["label"]) - set(CLASSES))
        if bad:
            raise SystemExit(f"{f.name} has unknown labels: {bad}")
        numeric = df[["img_w", "img_h"] + X_COLUMNS + Y_COLUMNS + Z_COLUMNS]
        if numeric.isna().any().any() or (df[["img_w", "img_h"]] <= 0).any().any():
            raise SystemExit(f"{f.name} has empty or invalid numbers; re-record that file.")
        frames.append(df[COLUMNS])
    data = pd.concat(frames, ignore_index=True)
    # A group is one recording session of one person. Evaluation never puts
    # frames of the same group in both training and test data.
    data["group"] = data["person"] + "/" + data["session"]
    return data
