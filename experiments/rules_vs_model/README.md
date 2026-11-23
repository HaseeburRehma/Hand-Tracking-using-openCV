# Rules versus a trained model: a follow-up experiment (November 2026)

A follow-up to the 2024 final-year project *3D Model Simulation Using Hand Gestures*.
It is not part of the original project or its grade.

## The question

In the 2024 project, MediaPipe found 21 hand landmarks and rules we wrote by hand
decided which gesture was shown. This experiment asks: **on the same recorded
landmarks, does a small trained classifier recognise the five driving gestures
better than those hand-written rules?**

| Gesture | Command |
|---|---|
| thumb up | move straight |
| index finger up | turn right |
| middle finger up | turn left |
| little finger up | move back |
| closed fist (none) | stop |

## Result (November 2026)

7,500 frames of one person's hand, recorded in three sessions; every method tested on a session it had not seen.

| Method | Accuracy, all test frames | Mean of the three sessions |
|---|---|---|
| Hand-written rules (2024) | 75.8% | 76.4% |
| Logistic regression | 92.5% | 89.9% |
| k-nearest neighbours | 84.2% | 83.7% |
| Random forest | 92.5% | 92.2% |

The rules recognise the thumb, index and little finger on 99–100% of frames, the middle finger on 80% and a
closed fist on 0.7%. Full tables: `results/summary.md`. Two-page report:
`report/Rules_vs_Model_Report_Haseeb_Ur_Rehman.pdf`.

## What is in this folder

| File | What it does |
|---|---|
| `record.py` | Records labelled hand landmarks from the webcam into `data/` |
| `rules_baseline.py` | The original `recognize_gesture` rules, copied unchanged from `CarTracking.py` |
| `features.py` | Makes landmarks independent of hand position and size |
| `evaluate.py` | Scores the rules and three classifiers; writes `results/` |
| `make_report.py` | Builds `report/report.html` from `results/metrics.json` |
| `data/` | The recorded landmark files (no images, only coordinates) |
| `tests/` | Tests, including drawn (synthetic) hands used only for testing |
| `models/` | Put Google's MediaPipe Hand Landmarker model here (see Setup; not stored in this repository) |

## Setup (once)

Python 3.11 is needed; the MediaPipe library does not run on Python 3.14.

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Download the hand-landmark model from Google's official storage and save it as `models/hand_landmarker.task`:
`https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`

For the camera-free self-test, also save Google's sample photo as `tests/assets/sample_hands.jpg`:
`https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg`
(without it, the tests that need a photo are skipped).

Check that everything works, without a camera:

```bash
.venv/bin/python record.py --selftest
```

```bash
.venv/bin/python -m unittest discover -s tests
```

## Step 1: record data

```bash
.venv/bin/python record.py --person haseeb --session s1 --condition normal
```

The first time, macOS asks to allow camera access for the terminal.

For each gesture: press **SPACE**, wait for the 3-second countdown, then hold the
gesture until the counter reaches 300. While recording, move the hand slowly
(a little closer, farther, slightly tilted) so the frames are not all identical.
**S** skips a gesture, **Q** saves and quits.

Recording plan (about two hours in total):

| Session | When and where | Conditions to record |
|---|---|---|
| `s1` | Day 1, desk, daylight | `normal`, `near`, `far` |
| `s2` | Day 2, different room or evening light | `normal`, `dim`, `tilted` |
| `s3` | Day 3, different background | `normal`, `far` |
| second person | any day | `normal` (use their own `--person` name) |

Rules that keep the result honest:
- A **session** is one sitting. Use a new `--session` name every time you sit down again.
- Do not delete frames that look bad. Bad frames are part of the result.
- Record all five gestures in every file.
- Write down anything unusual (glasses glare, sleeve covering the wrist) for the report.

## Step 2: evaluate

```bash
.venv/bin/python evaluate.py
```

It prints the accuracy of each method and writes:
- `results/summary.md`: all tables
- `results/metrics.json`: the raw numbers
- `results/confusion_*.png`: which gestures each method confuses
- `results/learning_curve.png`: accuracy against the amount of training data

## How the comparison is kept fair

- **Unseen sessions.** A model is always tested on a session it was not trained on
  (leave-one-session-out). Neighbouring video frames are almost identical, so a
  random split would let a model memorise the test data.
- **Same frames.** The rules are scored on exactly the same test frames.
- **Rules unchanged.** The baseline is the 2024 code as written, thresholds included.
- **Repeatable.** Fixed random seed; the same data gives the same numbers.

## Known limits (state these in the report)

- Few people and sessions: results describe this data, not all users.
- The landmarks come from the current MediaPipe Hand Landmarker. The 2024
  project used the older MediaPipe Hands interface with `model_complexity=0`.
- A recorded gesture is the one the person was asked to show; frames in which
  the hand was moving between poses carry the same label.
- Speed is measured on landmarks only; finding the landmarks takes far longer
  than either the rules or a classifier.
- The "Half Up" rules compare pixel distances with thresholds of 0.1 and 0.05,
  so they can fire only when two fingertips are less than a twentieth of a pixel apart horizontally.

## Authorship

Recorded and run by Haseeb Ur Rehman. The scripts were written with the help of
an AI coding assistant; the original rules are from the 2024 team project.
