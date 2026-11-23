"""Tests for the experiment code. Run from the project folder:

    .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import evaluate  # noqa: E402
import rules_baseline  # noqa: E402
import synthetic  # noqa: E402
from common import CLASSES, COLUMNS, load_data  # noqa: E402
from features import to_features  # noqa: E402


def one_session(seed=0, frames=20):
    return synthetic.make_session("p", "s", "normal", 120, 0, frames, seed)


class Features(unittest.TestCase):
    def test_shape_and_finite(self):
        X = to_features(one_session())
        self.assertEqual(X.shape, (20 * len(CLASSES), 42))
        self.assertTrue(np.isfinite(X).all())

    def test_same_hand_moved_and_resized_gives_same_features(self):
        df = one_session()
        moved = df.copy()
        xs = [f"x{i}" for i in range(21)]
        ys = [f"y{i}" for i in range(21)]
        # Shift in the picture and zoom by 0.5 around the origin.
        moved[xs] = df[xs] * 0.5 + 0.1
        moved[ys] = df[ys] * 0.5 + 0.05
        np.testing.assert_allclose(to_features(df), to_features(moved), atol=1e-4)

    def test_degenerate_hand_does_not_divide_by_zero(self):
        df = one_session().iloc[:1].copy()
        for i in range(21):
            df[f"x{i}"], df[f"y{i}"] = 0.5, 0.5
        self.assertTrue(np.isfinite(to_features(df)).all())


class Rules(unittest.TestCase):
    def tips(self, highest, gap=40.0):
        """Fingertip pixel coordinates with one finger clearly highest."""
        order = ["thumb", "index", "middle", "ring", "pinky"]
        ys = [300.0] * 5
        ys[order.index(highest)] = 100.0
        xs = [100.0 + gap * i for i in range(5)]
        return ys + xs

    def test_each_finger_is_recognised(self):
        for finger, cls in [("thumb", "thumb"), ("index", "index"), ("middle", "middle"), ("pinky", "pinky")]:
            out = rules_baseline.recognize_gesture(*self.tips(finger))
            self.assertEqual(rules_baseline.to_class(out), cls, out)

    def test_ring_finger_or_flat_hand_is_no_gesture(self):
        self.assertEqual(rules_baseline.recognize_gesture(*self.tips("ring")), "No Gesture")
        flat = [200.0] * 5 + [100.0, 140.0, 180.0, 220.0, 260.0]
        self.assertEqual(rules_baseline.recognize_gesture(*flat), "No Gesture")

    def test_half_up_needs_tips_closer_than_a_twentieth_of_a_pixel(self):
        # One pixel apart: full gesture. 0.01 pixel apart: "Half Up".
        self.assertNotIn("Half", rules_baseline.recognize_gesture(*self.tips("index", gap=1.0)))
        self.assertIn("Half", rules_baseline.recognize_gesture(*self.tips("index", gap=0.01)))

    def test_half_up_counts_as_its_finger(self):
        self.assertEqual(rules_baseline.to_class("Index Finger Half Up (Partial Right Turn)"), "index")
        self.assertEqual(rules_baseline.to_class("No Gesture"), "none")

    def test_predict_returns_one_class_per_frame(self):
        df = one_session()
        cls, raw = rules_baseline.predict(df)
        self.assertEqual(len(cls), len(df))
        self.assertTrue(set(cls) <= set(CLASSES))
        self.assertEqual(len(raw), len(df))


class Splits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_session_is_in_both_train_and_test(self):
        synthetic.write_dataset(self.dir, frames=10)
        data = load_data(self.dir).reset_index(drop=True)
        folds, text = evaluate.make_folds(data)
        self.assertEqual(len(folds), 3)
        self.assertIn("leave-one-session-out", text)
        for tr, te, _ in folds:
            self.assertFalse(set(data.loc[tr, "group"]) & set(data.loc[te, "group"]))
            self.assertEqual(len(set(tr) & set(te)), 0)

    def test_single_session_uses_time_split_with_gap(self):
        one_session(frames=50).to_csv(self.dir / "p_s_normal.csv", index=False)
        data = load_data(self.dir).reset_index(drop=True)
        folds, text = evaluate.make_folds(data)
        self.assertEqual(len(folds), 1)
        self.assertIn("ONE session", text)
        tr, te, _ = folds[0]
        self.assertEqual(len(set(tr) & set(te)), 0)
        for label in CLASSES:
            t_train = data.loc[tr][data.loc[tr, "label"] == label]["t_ms"].max()
            t_test = data.loc[te][data.loc[te, "label"] == label]["t_ms"].min()
            self.assertLess(t_train, t_test)


class Loading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_folder_is_an_error(self):
        with self.assertRaises(SystemExit):
            load_data(self.dir)

    def test_unknown_label_is_an_error(self):
        df = one_session()
        df.loc[0, "label"] = "peace"
        df.to_csv(self.dir / "bad.csv", index=False)
        with self.assertRaises(SystemExit):
            load_data(self.dir)

    def test_empty_number_is_an_error(self):
        df = one_session()
        df.loc[3, "x4"] = float("nan")
        df.to_csv(self.dir / "bad.csv", index=False)
        with self.assertRaises(SystemExit):
            load_data(self.dir)

    def test_missing_column_is_an_error(self):
        one_session().drop(columns=["x5"]).to_csv(self.dir / "bad.csv", index=False)
        with self.assertRaises(SystemExit):
            load_data(self.dir)


class EndToEnd(unittest.TestCase):
    def test_full_run_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir, res1, res2 = Path(tmp) / "data", Path(tmp) / "r1", Path(tmp) / "r2"
            synthetic.write_dataset(data_dir, frames=40)
            r1 = evaluate.run(data_dir, res1)
            r2 = evaluate.run(data_dir, res2)
            self.assertEqual(r1["frames"], 3 * 5 * 40)
            self.assertEqual(r1["test_frames"], r1["frames"])
            for name in ["rules", "logistic_regression", "k_nearest_neighbours", "random_forest"]:
                m = r1["metrics"][name]
                self.assertTrue(0.0 <= m["accuracy"] <= 1.0)
                self.assertEqual(set(m["recall_per_class"]), set(CLASSES))
            # Same data, same seed: same scores.
            self.assertEqual(r1["metrics"], r2["metrics"])
            self.assertEqual(r1["learning_curve_logistic_regression"], r2["learning_curve_logistic_regression"])
            for f in ["summary.md", "metrics.json", "learning_curve.png", "confusion_rules.png",
                      "confusion_logistic_regression.png"]:
                self.assertTrue((res1 / f).stat().st_size > 0, f)
            json.loads((res1 / "metrics.json").read_text())
            # Clean drawn hands must be easy for a trained model.
            self.assertGreater(r1["metrics"]["logistic_regression"]["accuracy"], 0.9)


SAMPLE_PHOTO = ROOT / "tests" / "assets" / "sample_hands.jpg"
MODEL_FILE = ROOT / "models" / "hand_landmarker.task"
NEEDS_DOWNLOADS = unittest.skipUnless(
    SAMPLE_PHOTO.exists() and MODEL_FILE.exists(),
    "needs the model file and the sample photo (see README, Setup)")


class Recorder(unittest.TestCase):
    def run_record(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "record.py"), *args],
                              capture_output=True, text=True, cwd=ROOT, timeout=120)

    @NEEDS_DOWNLOADS
    def test_selftest_finds_a_hand_in_the_sample_photo(self):
        out = self.run_record("--selftest")
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        self.assertIn("Self-test OK", out.stdout)

    def test_names_that_could_escape_the_data_folder_are_rejected(self):
        for bad in ["../evil", "a/b", "x" * 40, "name with space", ""]:
            out = self.run_record("--person", bad, "--session", "s1")
            self.assertNotEqual(out.returncode, 0, bad)
        self.assertFalse((ROOT.parent / "evil_s1_normal.csv").exists())

    def test_person_and_session_are_required(self):
        self.assertNotEqual(self.run_record().returncode, 0)

    def test_csv_header_matches_loader(self):
        self.assertEqual(len(COLUMNS), 9 + 63)
        self.assertEqual(len(set(COLUMNS)), len(COLUMNS))


class RecorderLoop(unittest.TestCase):
    """Run the real recording loop with a fake camera that shows the sample photo."""

    @NEEDS_DOWNLOADS
    def test_hands_free_recording_writes_a_loadable_file(self):
        import argparse
        from unittest import mock

        import cv2

        import record

        photo = cv2.imread(str(ROOT / "tests" / "assets" / "sample_hands.jpg"))

        class FakeCamera:
            def __init__(self, *_):
                pass

            def isOpened(self):
                return True

            def read(self):
                return True, photo.copy()

            def release(self):
                pass

        keys = iter([ord(" ")])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(record, "DATA_DIR", Path(tmp)), \
                mock.patch.object(record.cv2, "VideoCapture", FakeCamera), \
                mock.patch.object(record.cv2, "imshow", lambda *a: None), \
                mock.patch.object(record.cv2, "namedWindow", lambda *a: None), \
                mock.patch.object(record.cv2, "setWindowProperty", lambda *a: None), \
                mock.patch.object(record.cv2, "destroyAllWindows", lambda *a: None), \
                mock.patch.object(record.cv2, "waitKey", lambda *_: next(keys, -1)):
            args = argparse.Namespace(person="tester", session="s1", condition="photo", frames=4,
                                      rounds=1, camera=0, append=False, auto=True, countdown=0,
                                      idle_timeout=5)
            record.record(args)
            data = load_data(Path(tmp))
            self.assertEqual(len(data), 4 * len(CLASSES))
            self.assertEqual(data["label"].value_counts().to_dict(), {c: 4 for c in CLASSES})
            self.assertTrue(np.isfinite(to_features(data)).all())
            # A second run must not overwrite the file.
            with self.assertRaises(SystemExit):
                record.record(args)

    @NEEDS_DOWNLOADS
    def test_hands_free_mode_gives_up_if_nobody_starts(self):
        import argparse
        from unittest import mock

        import cv2

        import record

        photo = cv2.imread(str(ROOT / "tests" / "assets" / "sample_hands.jpg"))

        class FakeCamera:
            def __init__(self, *_):
                pass

            def isOpened(self):
                return True

            def read(self):
                return True, photo.copy()

            def release(self):
                pass

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(record, "DATA_DIR", Path(tmp)), \
                mock.patch.object(record.cv2, "VideoCapture", FakeCamera), \
                mock.patch.object(record.cv2, "imshow", lambda *a: None), \
                mock.patch.object(record.cv2, "namedWindow", lambda *a: None), \
                mock.patch.object(record.cv2, "setWindowProperty", lambda *a: None), \
                mock.patch.object(record.cv2, "destroyAllWindows", lambda *a: None), \
                mock.patch.object(record.cv2, "waitKey", lambda *_: -1):
            args = argparse.Namespace(person="tester", session="s1", condition="photo", frames=4,
                                      rounds=1, camera=0, append=False, auto=True, countdown=0,
                                      idle_timeout=0.5)
            record.record(args)
            self.assertFalse((Path(tmp) / "tester_s1_photo.csv").exists())


if __name__ == "__main__":
    unittest.main()
