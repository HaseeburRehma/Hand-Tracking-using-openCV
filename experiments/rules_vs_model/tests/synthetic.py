"""Make SYNTHETIC landmark files for testing the pipeline only.

These are drawn hands, not recordings. They exist so the code can be tested
without a camera. Never copy them into data/ and never report results on them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CLASSES, COLUMNS  # noqa: E402

# Finger base positions (relative to wrist, hand-size units; y grows downward).
BASES = {"thumb": (-0.45, -0.35), "index": (-0.22, -1.0), "middle": (0.0, -1.05),
         "ring": (0.2, -1.0), "pinky": (0.4, -0.9)}
DIRECTIONS = {"thumb": (-0.75, -0.65), "index": (-0.08, -1.0), "middle": (0.0, -1.0),
              "ring": (0.08, -1.0), "pinky": (0.2, -0.98)}
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def hand(label: str, rng) -> np.ndarray:
    """21 (x, y) points in hand units: one raised finger, the rest curled."""
    pts = [(0.0, 0.0)]
    for finger in FINGERS:
        bx, by = BASES[finger]
        dx, dy = DIRECTIONS[finger]
        raised = finger == label
        # Raised: joints go outward. Curled: the tip folds back toward the palm.
        steps = [0.0, 0.35, 0.65, 0.95] if raised else [0.0, 0.25, 0.2, 0.05]
        for s in steps:
            pts.append((bx + dx * s, by + dy * s))
    return np.array(pts) + rng.normal(0, 0.02, (21, 2))


def make_session(person: str, session: str, condition: str, scale: float, tilt_deg: float,
                 frames: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w, h = 1280, 720
    rows = []
    t = 0
    for label in CLASSES:
        for _ in range(frames):
            pts = hand(label, rng)
            angle = np.deg2rad(tilt_deg + rng.normal(0, 4))
            rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            px = pts @ rot.T * scale * rng.normal(1, 0.05)
            px += np.array([w * rng.uniform(0.35, 0.65), h * rng.uniform(0.6, 0.8)])
            t += 33
            row = dict(person=person, session=session, condition=condition, label=label,
                       frame=t // 33, t_ms=t, img_w=w, img_h=h, handedness="Right")
            for i in range(21):
                row[f"x{i}"] = round(px[i, 0] / w, 6)
                row[f"y{i}"] = round(px[i, 1] / h, 6)
                row[f"z{i}"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]


def write_dataset(out_dir: Path, frames: int = 60) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = [("anna", "s1", "normal", 150, 0), ("anna", "s2", "far", 80, 10), ("ben", "s1", "tilted", 130, -25)]
    for k, (person, session, condition, scale, tilt) in enumerate(plan):
        df = make_session(person, session, condition, scale, tilt, frames, seed=k)
        df.to_csv(out_dir / f"{person}_{session}_{condition}.csv", index=False)
