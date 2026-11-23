"""Turn raw landmarks into model inputs.

Each hand is described relative to itself, so the position of the hand in the
picture and its distance from the camera do not matter:
  1. convert x, y to pixels (so the image's aspect ratio is respected),
  2. move the wrist (landmark 0) to the origin,
  3. divide by the hand size (wrist to middle-finger knuckle, landmark 9).
Result: 42 numbers per frame (x and y of 21 landmarks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import N_LANDMARKS, X_COLUMNS, Y_COLUMNS

WRIST, MIDDLE_MCP = 0, 9


def to_features(df: pd.DataFrame) -> np.ndarray:
    x = df[X_COLUMNS].to_numpy(dtype=float) * df["img_w"].to_numpy(dtype=float)[:, None]
    y = df[Y_COLUMNS].to_numpy(dtype=float) * df["img_h"].to_numpy(dtype=float)[:, None]
    x = x - x[:, [WRIST]]
    y = y - y[:, [WRIST]]
    size = np.hypot(x[:, MIDDLE_MCP], y[:, MIDDLE_MCP])
    size = np.where(size < 1e-9, 1.0, size)  # degenerate frame: leave unscaled
    feats = np.concatenate([x / size[:, None], y / size[:, None]], axis=1)
    assert feats.shape[1] == 2 * N_LANDMARKS
    return feats
