# 3D Model Simulation Using Hand Gestures

Control a 3D scene in Unity with bare-hand gestures seen by an ordinary webcam. No glove, marker or depth sensor is needed.

This was my final-year project for the BSc in Computer Science at the Institute of Space Technology (KICSIT campus), Pakistan. It was built by a team of three; I was the project lead. Supervisor: Dr. Saif ul Islam. The project ran from April 2023 to February 2024 and received an A in both Design Project I and Design Project II.

## How it works

```
webcam ──> OpenCV ──> MediaPipe Hands ──> gesture rules ──> UDP (127.0.0.1:5052) ──> Unity (C#)
            frames      21 landmarks       one command                              moves the model
```

1. **Capture.** OpenCV reads frames from the webcam.
2. **Landmarks.** MediaPipe Hands returns 21 landmarks for the hand in each frame.
3. **Gesture recognition.** Hand-written rules compare the positions of the five fingertips and decide which finger is raised, fully or partly.
4. **Transport.** Python sends the result to Unity as a UDP message on the local machine.
5. **3D scene.** A C# script receives the messages on a background thread, and the scene reacts.

## Two modes

### 1. Drive a car with gestures (`CarTracking.py`)

| Gesture | Command | Key sent |
|---|---|---|
| Thumb up | Move forward | W |
| Index finger up | Turn right | D |
| Middle finger up | Turn left | A |
| Little finger up | Move back | S |
| No hand or no raised finger | Stop | all keys released |

A half-raised finger is recognised as the "partial" version of the same command. A command is sent only when the gesture changes, so the same gesture is not sent again on every frame. The recognised gesture is sent over UDP and also pressed as a W/A/S/D key (`keyinput.py`), which drives the car in the Unity city scene.

### 2. Show the hand in 3D (`main.py`)

All 21 landmark coordinates are sent to Unity on every frame. `HandTracking.cs` places 21 point objects at those coordinates, so the hand appears and moves in the 3D scene.

## Files

| File | What it does |
|---|---|
| `CarTracking.py` | Gesture recognition and car control (mode 1) |
| `keyinput.py` | Presses and releases W/A/S/D through the Windows `SendInput` API |
| `main.py` | Sends the 21 hand landmarks to Unity (mode 2) |
| `hand tracking.py` | Hand detector module (see Credits) |
| `UDPReceive.cs` | Unity script: receives UDP messages on a background thread |
| `HandTracking.cs` | Unity script: moves the 21 hand points |
| `*.mat`, `*.physicMaterial`, `*.meta` | Unity materials and settings used in the scenes |

The full Unity project (city scene, car model, render-pipeline settings) is not included in this repository.

## Requirements

- Python 3 with `opencv-python`, `mediapipe` and `cvzone`
- A webcam
- Unity (the project used the Universal Render Pipeline)
- Windows for mode 1, because `keyinput.py` uses the Windows `SendInput` API

## Run it

```bash
pip install opencv-python mediapipe cvzone
```

Mode 1: open the Unity car scene, press Play, then run:

```bash
python CarTracking.py
```

Mode 2: in a Unity scene, add `UDPReceive.cs` (port 5052) and `HandTracking.cs` to an object, assign 21 point objects to `handPoints`, press Play, then run:

```bash
python main.py
```

Press `q` in the camera window to stop mode 1.

## Limits

- The gesture rules and their thresholds were set by hand. Recognition accuracy and delay were not measured with numbers.
- Landmark detection comes from the pre-trained MediaPipe model. No model was trained in this project.
- Testing was functional: each gesture was shown to the camera and the expected action was checked in Unity.
- Mode 1 works on Windows only.

## Follow-up experiment (November 2026)

The limits above say the rules were never measured. In November 2026 I measured them against small trained
classifiers on 7,500 recorded frames, each method tested on a session it had not seen: the rules recognised 76% of
gestures, a random forest 92%. The rules are reliable for the thumb, index and little finger but almost never
recognise a closed fist. Code, data and a two-page report: [`experiments/rules_vs_model`](experiments/rules_vs_model).
This follow-up is not part of the graded 2024 project.

## Credits

- [MediaPipe Hands](https://arxiv.org/abs/2006.10214) (Zhang et al., 2020) for hand landmark detection.
- `hand tracking.py` is the `HandTrackingModule` from the open-source [cvzone](https://github.com/cvzone/cvzone) library, kept here for reference. `main.py`, `UDPReceive.cs` and `HandTracking.cs` follow the cvzone author's example for streaming hand landmarks into Unity over UDP.
- `keyinput.py` is adapted from a widely shared Windows `SendInput` example for sending keyboard scan codes.
- The gesture rules, the gesture-to-command mapping and the car control in `CarTracking.py` are the team's own work.
