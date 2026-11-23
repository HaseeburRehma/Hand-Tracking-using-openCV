"""Build report/report.html from results/metrics.json, so every number in the
report is the number the evaluation produced.

    .venv/bin/python make_report.py

Then print the HTML to PDF (see README).
"""
from __future__ import annotations

import json

from common import CLASSES, RESULTS_DIR, ROOT

NAMES = {
    "rules": "Hand-written rules (2024)",
    "logistic_regression": "Logistic regression",
    "k_nearest_neighbours": "k-nearest neighbours (k = 5)",
    "random_forest": "Random forest (200 trees)",
}
GESTURES = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "pinky": "Little finger", "none": "Closed fist"}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build() -> str:
    r = json.loads((RESULTS_DIR / "metrics.json").read_text())
    m, folds, mean = r["metrics"], r["accuracy_per_fold"], r["mean_session_accuracy"]
    sessions = list(folds["rules"])

    main_rows = "".join(
        f"<tr><td>{NAMES[k]}</td><td>{pct(m[k]['accuracy'])}</td><td>{pct(mean[k])}</td>"
        + "".join(f"<td>{pct(folds[k][s])}</td>" for s in sessions)
        + f"<td>{m[k]['macro_f1']:.3f}</td><td>{r['ms_per_frame'][k]:.2f}</td></tr>"
        for k in NAMES)
    gesture_rows = "".join(
        f"<tr><td>{GESTURES[c]}</td>" + "".join(f"<td>{pct(m[k]['recall_per_class'][c])}</td>" for k in NAMES) + "</tr>"
        for c in CLASSES)
    curve_rows = "".join(
        f"<tr><td>{c['train_frames']}</td><td>{pct(c['accuracy'])}</td></tr>"
        for c in r["learning_curve_logistic_regression"])
    session_heads = "".join(f"<th>Test on {s.split('/')[1]}</th>" for s in sessions)
    method_heads = "".join(f"<th>{NAMES[k]}</th>" for k in NAMES)
    best = max((k for k in NAMES if k != "rules"), key=lambda k: mean[k])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Hand-written rules versus trained classifiers for gesture recognition</title>
<style>
@page{{size:A4;margin:17mm 18mm}}
body{{font-family:"Helvetica Neue",Arial,sans-serif;color:#222;font-size:10pt;line-height:1.42;margin:0}}
h1{{font-size:15pt;margin:0 0 1mm;color:#0b4a8b}}
.sub{{font-size:9.4pt;color:#444;margin:0 0 3mm}}
h2{{font-size:10.6pt;color:#0b4a8b;margin:3.6mm 0 1mm;text-transform:uppercase;letter-spacing:.3px;break-after:avoid}}
p{{margin:0 0 1.5mm}}ul{{margin:0 0 1.5mm;padding-left:5mm}}li{{margin-bottom:.6mm}}
table{{border-collapse:collapse;width:100%;margin:1mm 0 2.5mm;font-size:8.8pt;break-inside:avoid}}
th,td{{border-bottom:1px solid #c9d3e0;padding:1.1mm 1.2mm;text-align:right}}
th{{color:#0b4a8b;font-weight:600}}td:first-child,th:first-child{{text-align:left}}
.figs{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:3mm;align-items:start;break-inside:avoid}}
figure{{margin:0}}figure img{{width:100%;display:block}}figcaption{{font-size:8.2pt;color:#444;line-height:1.3}}
.note{{font-size:8.6pt;color:#555;border-top:1px solid #d5dde8;margin-top:3mm;padding-top:1.5mm}}
code{{font-family:Menlo,monospace;font-size:8.6pt}}a{{color:#0b4a8b}}
</style></head><body>
<h1>Hand-written rules versus trained classifiers for gesture recognition</h1>
<p class="sub">A follow-up experiment to the 2024 final-year project <i>3D Model Simulation Using Hand Gestures</i> ·
Haseeb Ur Rehman · November 2026 ·
<a href="https://github.com/HaseeburRehma/Hand-Tracking-using-openCV/tree/main/experiments/rules_vs_model">code and data</a></p>

<h2>Question</h2>
<p>In the 2024 project, MediaPipe located 21 hand landmarks and rules written by hand decided which of five driving
gestures was shown. We set those rules by trial and error and never measured them. This experiment asks: on the same
recorded landmarks, does a small trained classifier recognise the gestures better than the rules, and how much data
does it need?</p>

<h2>Data</h2>
<p>I recorded {r['frames']:,} frames of my own hand with a laptop webcam (1280 × 720): {r['frames_per_class']['thumb']:,}
per gesture, in {len(sessions)} sessions on 22 and 23 November 2026. A session is one sitting; the three were recorded at
different times of day and distances from the camera (median hand size between 191 and 333 pixels). Each frame stores the 21
landmarks returned by the MediaPipe Hand Landmarker and the gesture I was asked to show. Gestures: thumb up (move
straight), index finger up (turn right), middle finger up (turn left), little finger up (move back), closed fist (stop).
No frame was removed.</p>

<h2>Methods</h2>
<ul>
<li><b>Baseline.</b> The function <code>recognize_gesture</code> from the 2024 code, unchanged, thresholds included.
It compares the heights of the five fingertips and names the highest one.</li>
<li><b>Trained classifiers.</b> Logistic regression, k-nearest neighbours and a random forest (scikit-learn, default
settings, fixed seed). Input: the x and y of all 21 landmarks, moved so the wrist is the origin and divided by the
hand size, so position and distance do not matter (42 numbers).</li>
<li><b>Test.</b> Leave-one-session-out: train on two sessions, test on the third, three times. Neighbouring video
frames are almost identical, so a random split would let a model memorise its test data. The rules need no training
and are scored on the same test frames.</li>
</ul>

<h2>Results</h2>
<table><tr><th>Method</th><th>Accuracy, all test frames</th><th>Mean of the three sessions</th>{session_heads}<th>Macro F1</th><th>ms per frame</th></tr>
{main_rows}</table>
<p>The first column pools all {r['test_frames']:,} test frames, so session 1, which holds three recordings, weighs
more. The second gives each session the same weight. I report both; they agree on the order of the methods.</p>

<table><tr><th>Share of frames recognised correctly</th>{method_heads}</tr>{gesture_rows}</table>

<div class="figs">
<figure><img src="../results/confusion_rules.png" alt=""><figcaption><b>Figure 1.</b> The rules: a closed fist is almost always read as a raised finger.</figcaption></figure>
<figure><img src="../results/confusion_{best}.png" alt=""><figcaption><b>Figure 2.</b> {NAMES[best]} on unseen sessions.</figcaption></figure>
<figure><img src="../results/learning_curve.png" alt=""><figcaption><b>Figure 3.</b> Logistic regression against the amount of training data (mean of the three sessions).</figcaption></figure>
</div>

<h2>What the results show</h2>
<ul>
<li><b>The rules are good where a gesture has a simple definition.</b> One clearly raised finger is recognised on
{pct(m['rules']['recall_per_class']['thumb'])}, {pct(m['rules']['recall_per_class']['index'])} and
{pct(m['rules']['recall_per_class']['pinky'])} of frames (thumb, index, little finger), better than any trained model
for the thumb.</li>
<li><b>The rules cannot say "no gesture".</b> They always name the highest fingertip, so a closed fist is recognised on
only {pct(m['rules']['recall_per_class']['none'])} of frames. This single gesture explains most of the gap.</li>
<li><b>The middle finger is the rules' second weak point</b> ({pct(m['rules']['recall_per_class']['middle'])}): it is
often read as the index finger, most often when the hand is close to the camera.</li>
<li><b>A trained model needs little data.</b> Logistic regression reaches {pct(r['learning_curve_logistic_regression'][1]['accuracy'])}
with {r['learning_curve_logistic_regression'][1]['train_frames']} training frames and does not improve after about 500.
More frames from the same sessions add little.</li>
<li><b>Not every model beats the rules everywhere.</b> k-nearest neighbours is worst on the fist among the trained
models, and logistic regression drops to {pct(folds['logistic_regression'][sessions[-1]])} on session 3, where the
rules reach {pct(folds['rules'][sessions[-1]])}.</li>
<li><b>The "Half Up" rules never mattered.</b> They compare distances in pixels with thresholds of 0.1 and 0.05, and
fired on {r['rules_half_outputs']} of {r['frames']:,} frames.</li>
<li><b>The rules are the fastest</b> ({r['ms_per_frame']['rules']:.2f} ms per frame); logistic regression takes
{r['ms_per_frame']['logistic_regression']:.2f} ms and the random forest {r['ms_per_frame']['random_forest']:.0f} ms. The time to
find the landmarks was not measured.</li>
</ul>

<h2>Limits</h2>
<ul>
<li>One person, one camera, three sittings over two days. The numbers describe this data, not other hands or users.</li>
<li>A frame's label is the gesture I was asked to show; frames where the hand was still settling keep that label.</li>
<li>The landmarks come from the current MediaPipe Hand Landmarker; the 2024 project used the older MediaPipe Hands
interface with <code>model_complexity=0</code>.</li>
<li>The classifiers use default settings. No tuning was done, so the test sessions were never used to choose a model.</li>
</ul>

<h2>Conclusion</h2>
<p>For this task the boundary between rules and a learned model follows how easily a gesture can be defined. Where the
definition is one comparison (which fingertip is highest), the hand-written rules are accurate, fast and need no data.
Where the definition is the absence of something (no finger raised), they fail, and a small classifier trained on a
few hundred frames does much better. Testing other people's hands is the next step.</p>

<p class="note">Reproduce: <code>evaluate.py</code> on the CSV files in <code>data/</code> gives these numbers (seed 0).
I recorded all data and ran the experiment. The scripts were written with the help of an AI coding assistant; the
rules are from the 2024 team project (three students, supervisor Dr. Saif ul Islam).</p>
</body></html>
"""


def main() -> None:
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    (out / "report.html").write_text(build())
    print(f"Written: {out / 'report.html'}")


if __name__ == "__main__":
    main()
