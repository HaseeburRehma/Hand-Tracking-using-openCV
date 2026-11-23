"""Compare the hand-written rules with trained classifiers on recorded data.

    .venv/bin/python evaluate.py

How the comparison is kept fair
- Test frames always come from a recording session the model never saw
  (leave-one-session-out). Neighbouring video frames are almost identical, so
  a random split would let the model memorise the test data.
- The rules need no training, so they are scored on exactly the same test
  frames, fold by fold.
- Everything is seeded; running it twice gives the same numbers.

Outputs go to results/: summary.md, metrics.json and PNG figures.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rules_baseline
from common import CLASSES, DATA_DIR, RESULTS_DIR, load_data
from features import to_features

SEED = 0
FRACTIONS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]


def make_models() -> dict:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "k_nearest_neighbours": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }


def make_folds(data: pd.DataFrame) -> tuple[list[tuple[np.ndarray, np.ndarray, str]], str]:
    """Return (train_idx, test_idx, name) folds and a description of the split."""
    groups = data["group"].to_numpy()
    names = sorted(set(groups))
    if len(names) >= 2:
        folds = [(np.flatnonzero(groups != g), np.flatnonzero(groups == g), g) for g in names]
        return folds, f"leave-one-session-out over {len(names)} sessions"
    # Only one session: train on the first 60% of each gesture's frames, skip
    # 10%, test on the last 30%. Weaker evidence than a new session.
    train, test = [], []
    for _, part in data.groupby("label", sort=False):
        idx = part.sort_values("t_ms").index.to_numpy()
        a, b = int(len(idx) * 0.6), int(len(idx) * 0.7)
        train.extend(idx[:a])
        test.extend(idx[b:])
    return [(np.array(train), np.array(test), names[0] + " (time split)")], (
        "ONE session only: time-ordered 60% train / 30% test split inside it "
        "(record more sessions for a real test)")


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "recall_per_class": {c: float(r) for c, r in zip(
            CLASSES, recall_score(y_true, y_pred, labels=CLASSES, average=None, zero_division=0))},
    }


def time_per_frame_ms(fn, n: int) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000 / max(n, 1)


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true gesture")
    ax.set_title(title, fontsize=10)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> dict:
    data = load_data(data_dir).reset_index(drop=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    X = to_features(data)
    y = data["label"].to_numpy()
    folds, split_text = make_folds(data)

    # Rules: no training, one prediction per frame.
    rule_pred, rule_raw = rules_baseline.predict(data)
    half_count = int(sum("Half" in t for t in rule_raw))

    test_idx_all = np.concatenate([te for _, te, _ in folds])
    predictions = {"rules": rule_pred[test_idx_all]}
    per_fold = {"rules": {}}
    for _, te, name in folds:
        per_fold["rules"][name] = float(accuracy_score(y[te], rule_pred[te]))

    for model_name in make_models():
        parts, per_fold[model_name] = [], {}
        for tr, te, name in folds:
            model = make_models()[model_name]
            model.fit(X[tr], y[tr])
            pred = model.predict(X[te])
            parts.append(pred)
            per_fold[model_name][name] = float(accuracy_score(y[te], pred))
        predictions[model_name] = np.concatenate(parts)

    y_test = y[test_idx_all]
    metrics = {name: scores(y_test, pred) for name, pred in predictions.items()}

    # Accuracy per recording condition (near, far, dim ...), on test frames.
    cond = data["condition"].to_numpy()[test_idx_all]
    per_condition = {
        name: {c: float(accuracy_score(y_test[cond == c], pred[cond == c])) for c in sorted(set(cond))}
        for name, pred in predictions.items()
    }

    # Speed: time to classify frames that already have landmarks.
    sample = data.iloc[test_idx_all[:2000]]
    Xs = X[test_idx_all[:2000]]
    speed = {"rules": time_per_frame_ms(lambda: rules_baseline.predict(sample), len(sample))}
    tr0 = folds[0][0]
    for model_name in make_models():
        model = make_models()[model_name].fit(X[tr0], y[tr0])
        speed[model_name] = time_per_frame_ms(lambda m=model: [m.predict(r[None, :]) for r in Xs], len(Xs))

    # Learning curve: how much training data does the simplest model need?
    rng = np.random.default_rng(SEED)
    curve = []
    for frac in FRACTIONS:
        accs, sizes = [], []
        for tr, te, _ in folds:
            n = max(int(len(tr) * frac), len(CLASSES) * 2)
            for _ in range(20):  # resample until every class is present
                sub = rng.choice(tr, size=min(n, len(tr)), replace=False)
                if len(set(y[sub])) == len(set(y[tr])):
                    break
            model = make_models()["logistic_regression"].fit(X[sub], y[sub])
            accs.append(accuracy_score(y[te], model.predict(X[te])))
            sizes.append(len(sub))
        curve.append({"fraction": frac, "train_frames": int(np.mean(sizes)), "accuracy": float(np.mean(accs))})

    report = {
        "frames": int(len(data)),
        "test_frames": int(len(test_idx_all)),
        "sessions": sorted(set(data["group"])),
        "people": sorted(set(data["person"])),
        "conditions": sorted(set(data["condition"])),
        "frames_per_class": {c: int((y == c).sum()) for c in CLASSES},
        "split": split_text,
        "metrics": metrics,
        "accuracy_per_fold": per_fold,
        # Unweighted mean over held-out sessions: no session counts more
        # because it happens to hold more frames.
        "mean_session_accuracy": {k: float(np.mean(list(v.values()))) for k, v in per_fold.items()},
        "accuracy_per_condition": per_condition,
        "ms_per_frame": speed,
        "rules_half_outputs": half_count,
        "learning_curve_logistic_regression": curve,
        "seed": SEED,
    }
    (results_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    for name, pred in predictions.items():
        plot_confusion(y_test, pred, f"{name}  (accuracy {metrics[name]['accuracy']:.1%})",
                       results_dir / f"confusion_{name}.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot([c["train_frames"] for c in curve], [c["accuracy"] for c in curve], marker="o",
            label="logistic regression")
    ax.axhline(metrics["rules"]["accuracy"], color="black", linestyle="--", label="hand-written rules")
    ax.set_xscale("log")
    ax.set_xlabel("training frames (log scale)")
    ax.set_ylabel("accuracy on unseen session")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "learning_curve.png", dpi=160)
    plt.close(fig)

    write_summary(report, results_dir / "summary.md")
    return report


def write_summary(r: dict, path: Path) -> None:
    lines = [
        "# Rules versus trained models: results",
        "",
        f"- Frames recorded: {r['frames']} ({', '.join(f'{c} {n}' for c, n in r['frames_per_class'].items())})",
        f"- People: {', '.join(r['people'])} · sessions: {len(r['sessions'])} · conditions: {', '.join(r['conditions'])}",
        f"- Split: {r['split']}; {r['test_frames']} test frames",
        "",
        "| Method | Accuracy | Macro F1 | " + " | ".join(f"recall {c}" for c in CLASSES) + " | ms per frame |",
        "|---|---|---|" + "---|" * len(CLASSES) + "---|",
    ]
    for name, m in r["metrics"].items():
        rec = " | ".join(f"{m['recall_per_class'][c]:.1%}" for c in CLASSES)
        lines.append(f"| {name} | {m['accuracy']:.1%} | {m['macro_f1']:.3f} | {rec} | {r['ms_per_frame'][name]:.3f} |")
    lines += ["", "## Accuracy per held-out session", "",
              "| Method | " + " | ".join(r["accuracy_per_fold"]["rules"]) + " |",
              "|---|" + "---|" * len(r["accuracy_per_fold"]["rules"])]
    for name, folds in r["accuracy_per_fold"].items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.1%}" for v in folds.values()) + " |")
    lines += ["", "Mean over sessions (each session counts equally): "
              + " · ".join(f"{k} {v:.1%}" for k, v in r["mean_session_accuracy"].items())]
    lines += ["", "## Accuracy per recording condition", "",
              "| Method | " + " | ".join(r["conditions"]) + " |",
              "|---|" + "---|" * len(r["conditions"])]
    for name, conds in r["accuracy_per_condition"].items():
        lines.append(f"| {name} | " + " | ".join(f"{conds[c]:.1%}" for c in r["conditions"]) + " |")
    lines += ["", "## Learning curve (logistic regression)", "",
              "| Training frames | Accuracy |", "|---|---|"]
    lines += [f"| {c['train_frames']} | {c['accuracy']:.1%} |" for c in r["learning_curve_logistic_regression"]]
    lines += ["",
              f"The rules returned a \"Half Up\" label on {r['rules_half_outputs']} of {r['frames']} frames.",
              "",
              "Figures: confusion_*.png, learning_curve.png. Raw numbers: metrics.json.",
              ""]
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA_DIR)
    p.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = p.parse_args()
    report = run(args.data, args.results)
    print(f"Split: {report['split']}")
    for name, m in report["metrics"].items():
        print(f"  {name:22s} accuracy {m['accuracy']:.1%}   macro F1 {m['macro_f1']:.3f}")
    print(f"Written: {args.results}/summary.md")


if __name__ == "__main__":
    main()
