"""The hand-written gesture rules of the 2024 project, used as the baseline.

`recognize_gesture` is copied without changes from CarTracking.py in
github.com/HaseeburRehma/Hand-Tracking-using-openCV, including its two
thresholds. Do not "fix" it here: the experiment measures the rules as they
were written.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- copied from CarTracking.py (unchanged) ---------------------------------
gesture_threshold = 0.1  # Adjusted gesture_threshold
partial_gesture_threshold = 0.05  # Adjusted partial_gesture_threshold


def recognize_gesture(thumb_y, index_y, middle_y, ring_y, pinky_y, thumb_x, index_x, middle_x, ring_x, pinky_x):
    is_thumb_up = thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y and abs(thumb_x - index_x) > gesture_threshold
    is_index_up = index_y < middle_y and index_y < ring_y and index_y < pinky_y and abs(index_x - middle_x) > gesture_threshold
    is_middle_up = middle_y < index_y and middle_y < ring_y and middle_y < pinky_y and abs(middle_x - index_x) > gesture_threshold
    is_pinky_up = pinky_y < index_y and pinky_y < middle_y and pinky_y < ring_y and abs(pinky_x - index_x) > gesture_threshold

    is_thumb_half_up = thumb_y < index_y and thumb_y < middle_y and thumb_y < ring_y and thumb_y < pinky_y and abs(thumb_x - index_x) < partial_gesture_threshold
    is_index_half_up = index_y < middle_y and index_y < ring_y and index_y < pinky_y and abs(index_x - middle_x) < partial_gesture_threshold
    is_middle_half_up = middle_y < index_y and middle_y < ring_y and middle_y < pinky_y and abs(middle_x - index_x) < partial_gesture_threshold
    is_pinky_half_up = pinky_y < index_y and pinky_y < middle_y and pinky_y < ring_y and abs(pinky_x - index_x) < partial_gesture_threshold

    if is_thumb_up:
        return "Thumb Up (Move Straight)"
    elif is_index_up:
        return "Index Finger Up (Turn Right)"
    elif is_middle_up:
        return "Middle Finger Up (Turn Left)"
    elif is_pinky_up:
        return "Pinky Finger Up (Move Back)"
    elif is_thumb_half_up:
        return "Thumb Half Up (Partial Acceleration)"
    elif is_index_half_up:
        return "Index Finger Half Up (Partial Right Turn)"
    elif is_middle_half_up:
        return "Middle Finger Half Up (Partial Left Turn)"
    elif is_pinky_half_up:
        return "Pinky Finger Half Up (Partial Move Back)"
    else:
        return "No Gesture"
# --- end of copied code ------------------------------------------------------

# MediaPipe fingertip indices: thumb, index, middle, ring, little finger.
TIPS = (4, 8, 12, 16, 20)


def to_class(rule_output: str) -> str:
    """Map the rule's text label to one of the five experiment classes.

    The "Half Up" variants count as their finger: the original program sent
    the same key press for both.
    """
    if rule_output.startswith("Thumb"):
        return "thumb"
    if rule_output.startswith("Index"):
        return "index"
    if rule_output.startswith("Middle"):
        return "middle"
    if rule_output.startswith("Pinky"):
        return "pinky"
    return "none"


def rule_output_for_row(row) -> str:
    """Run the rules on one recorded frame.

    The original code multiplied the normalised landmark by the image size
    (pixels) before calling the rules, so this does the same.
    """
    w, h = float(row["img_w"]), float(row["img_h"])
    ys = [float(row[f"y{i}"]) * h for i in TIPS]
    xs = [float(row[f"x{i}"]) * w for i in TIPS]
    return recognize_gesture(*ys, *xs)


def predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (class per frame, raw rule text per frame)."""
    raw = np.array([rule_output_for_row(r) for _, r in df.iterrows()], dtype=object)
    return np.array([to_class(t) for t in raw], dtype=object), raw
