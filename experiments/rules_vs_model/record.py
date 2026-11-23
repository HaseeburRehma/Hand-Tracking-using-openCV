"""Record labelled hand landmarks from the webcam.

Example:
    .venv/bin/python record.py --person haseeb --session s1 --condition normal

For each of the five gestures the window shows what to do. Press SPACE, wait
for the countdown, then hold the gesture while the frames are counted. Move the
hand a little while recording (closer, farther, slight tilt) so the frames are
not all the same.

Keys: SPACE start a gesture · S skip it · Q save and quit.
With --auto, press SPACE once; every gesture then starts after its countdown.

Each frame with a detected hand becomes one CSV row in data/. The image is
mirrored first, as in the original CarTracking.py.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from common import CLASSES, COLUMNS, DATA_DIR, INSTRUCTIONS, MODEL_PATH, N_LANDMARKS, ROOT

FONT = cv2.FONT_HERSHEY_SIMPLEX
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,30}$")


def make_landmarker(mode):
    if not MODEL_PATH.exists():
        sys.exit(f"Model file missing: {MODEL_PATH} (see README, 'Setup').")
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def row_from_result(result, meta: dict) -> dict | None:
    """Build one CSV row from a detection result, or None if no hand."""
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    if len(hand) != N_LANDMARKS:
        return None
    row = dict(meta)
    row["handedness"] = result.handedness[0][0].category_name if result.handedness else ""
    for i, lm in enumerate(hand):
        row[f"x{i}"] = round(lm.x, 6)
        row[f"y{i}"] = round(lm.y, 6)
        row[f"z{i}"] = round(lm.z, 6)
    return row


def selftest() -> None:
    """Check the detector on a bundled sample photo (no camera needed)."""
    path = ROOT / "tests" / "assets" / "sample_hands.jpg"
    image = cv2.imread(str(path))
    if image is None:
        sys.exit(f"Sample image missing: {path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with make_landmarker(vision.RunningMode.IMAGE) as landmarker:
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    meta = dict(person="selftest", session="s0", condition="photo", label="none",
                frame=0, t_ms=0, img_w=image.shape[1], img_h=image.shape[0])
    row = row_from_result(result, meta)
    if row is None:
        sys.exit("Self-test FAILED: no hand found in the sample photo.")
    assert set(row) == set(COLUMNS), "row columns do not match the CSV header"
    print(f"Self-test OK: 21 landmarks found, {row['handedness']} hand, wrist at "
          f"({row['x0']}, {row['y0']}).")


def draw_text(image, lines, colour=(255, 255, 255)) -> None:
    for i, text in enumerate(lines):
        y = 32 + i * 30
        cv2.putText(image, text, (12, y), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (12, y), FONT, 0.7, colour, 2, cv2.LINE_AA)


def record(args) -> None:
    for name, value in (("person", args.person), ("session", args.session), ("condition", args.condition)):
        if not SAFE_NAME.match(value):
            sys.exit(f"--{name} must be 1-30 letters, digits, '-' or '_' (got {value!r}).")
    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / f"{args.person}_{args.session}_{args.condition}.csv"
    if out.exists() and not args.append:
        sys.exit(f"{out.name} already exists. Use a new --session/--condition, or --append.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the camera. On macOS, allow camera access for your "
                 "terminal in System Settings > Privacy & Security > Camera.")

    new_file = not out.exists()
    counts = {c: 0 for c in CLASSES}
    start = time.monotonic()
    frame_no = 0
    queue = [c for _ in range(args.rounds) for c in CLASSES]
    state, label, countdown_end, quit_now = "wait", None, 0.0, False
    started, opened = False, time.monotonic()  # auto mode: one SPACE starts everything
    cv2.namedWindow("record gestures")
    try:
        cv2.setWindowProperty("record gestures", cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass

    with open(out, "a", newline="") as fh, make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        while not quit_now:
            if state == "wait" and label is None:
                if not queue:
                    break
                label, recorded = queue.pop(0), 0
            ok, image = cap.read()
            if not ok:
                continue
            image = cv2.flip(image, 1)  # selfie view, as in CarTracking.py
            h, w = image.shape[:2]
            t_ms = int((time.monotonic() - start) * 1000)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)
            frame_no += 1
            hand_found = bool(result.hand_landmarks)
            if hand_found:
                pts = result.hand_landmarks[0]
                for lm in pts:
                    cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
                # Hand size on screen (wrist to middle knuckle), to judge distance.
                size_px = ((pts[9].x - pts[0].x) ** 2 * w * w + (pts[9].y - pts[0].y) ** 2 * h * h) ** 0.5
                cv2.putText(image, f"hand size {size_px:.0f} px", (12, h - 16), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, f"hand size {size_px:.0f} px", (12, h - 16), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            if state == "wait" and args.auto and started:
                state, countdown_end = "countdown", time.monotonic() + args.countdown
            if state == "wait":
                if args.auto:
                    draw_text(image, ["Hands-free mode: press SPACE once when you are ready",
                                      f"First gesture: {INSTRUCTIONS[label]}", "Q = quit"])
                    if time.monotonic() - opened > args.idle_timeout:
                        print("Nobody pressed SPACE; stopping without recording.")
                        quit_now = True
                else:
                    draw_text(image, [f"Next: {INSTRUCTIONS[label]}",
                                      "SPACE = start   S = skip   Q = save and quit"])
            elif state == "countdown":
                left = countdown_end - time.monotonic()
                draw_text(image, [f"Get ready: {INSTRUCTIONS[label]}", f"{max(left, 0):.0f}"], (0, 255, 255))
                if left <= 0:
                    state = "record"
            elif state == "record":
                if hand_found:
                    meta = dict(person=args.person, session=args.session, condition=args.condition,
                                label=label, frame=frame_no, t_ms=t_ms, img_w=w, img_h=h)
                    row = row_from_result(result, meta)
                    if row is not None:
                        writer.writerow(row)
                        recorded += 1
                        counts[label] += 1
                draw_text(image, [f"RECORDING {label}: {recorded}/{args.frames}",
                                  "Move slowly: closer, farther, slight tilt"
                                  if hand_found else "No hand detected"], (0, 0, 255))
                if recorded >= args.frames:
                    fh.flush()
                    state, label = "wait", None

            cv2.imshow("record gestures", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                quit_now = True
            elif state == "wait" and label is not None and key == ord(" "):
                started = True
                state, countdown_end = "countdown", time.monotonic() + args.countdown
            elif state == "wait" and label is not None and key in (ord("s"), ord("S")):
                label = None

    cap.release()
    cv2.destroyAllWindows()
    if new_file and sum(counts.values()) == 0:
        out.unlink()  # nothing was recorded: do not leave an empty file behind
        print("No frames recorded; no file written.")
        return
    print(f"Saved to {out}")
    for c in CLASSES:
        print(f"  {c:7s} {counts[c]} frames")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--person", help="who is in front of the camera, e.g. haseeb")
    p.add_argument("--session", help="recording session, e.g. s1 (use a new one each day/setup)")
    p.add_argument("--condition", default="normal", help="e.g. normal, near, far, dim, tilted")
    p.add_argument("--frames", type=int, default=300, help="frames per gesture (default 300)")
    p.add_argument("--rounds", type=int, default=1, help="times to go through all five gestures")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--append", action="store_true", help="add to an existing file")
    p.add_argument("--auto", action="store_true",
                   help="hands-free: press SPACE once, then each gesture starts by itself")
    p.add_argument("--countdown", type=float, default=3, help="seconds to get ready before each gesture")
    p.add_argument("--idle-timeout", type=float, default=300,
                   help="auto mode: give up if SPACE is not pressed within this many seconds")
    p.add_argument("--selftest", action="store_true", help="test the detector on a sample photo and exit")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.person or not args.session:
        p.error("--person and --session are required")
    if args.frames < 1 or args.rounds < 1:
        p.error("--frames and --rounds must be at least 1")
    record(args)


if __name__ == "__main__":
    main()
