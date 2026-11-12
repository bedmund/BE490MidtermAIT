import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

import glassbreak as gb


class GlassbreakTests(unittest.TestCase):
    def test_mixture_annotations_override_folders_and_keep_devtest_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "meta"
            meta.mkdir()
            audio = root / "normal"  # Deliberately wrong folder: metadata is authoritative.
            audio.mkdir()
            ids = {}
            expected_labels = {}
            for split, count in (("devtrain", 20), ("devtest", 4)):
                rows = []
                for i in range(count):
                    name = f"mixture_{split}_glassbreak_{i}.wav"
                    (audio / name).touch()
                    ids[name] = len(ids)
                    if i % 2:
                        if i % 4 == 1:
                            # Exactly [3,6): neither adjacent window is positive.
                            rows.append(f"{name}\t3\t6\tglassbreak")
                            expected_labels[ids[name]] = [0, 1, 0]
                        else:
                            rows.append(f"{name}\t2.5\t3.5\tglassbreak")
                            expected_labels[ids[name]] = [1, 1, 0]
                    else:
                        rows.append(name)
                        expected_labels[ids[name]] = [0, 0, 0]
                (meta / f"event_list_{split}_glassbreak.csv").write_text("\n".join(rows))
            # An obsolete annotation must never be loaded.
            (meta / "event_list_devtrain_babycry.csv_old_dontuse").write_text("invalid")

            def fake_features(path):
                return np.full((3, 64, 94), ids[Path(path).name], dtype=np.float32)

            with patch.object(gb, "extract_features", side_effect=fake_features), \
                    contextlib.redirect_stdout(io.StringIO()):
                splits = gb.prepare_data_pipeline(root, {}, meta)
            groups = [set(x[:, 0, 0]) for x, _ in splits]
            self.assertEqual(groups[2], set(range(20, 24)))
            self.assertTrue(groups[0].isdisjoint(groups[1]))
            self.assertEqual(groups[0] | groups[1], set(range(20)))
            for features, labels in splits:
                for i in range(0, len(labels), 3):
                    expected = expected_labels[features[i, 0, 0]]
                    np.testing.assert_array_equal(labels[i:i + 3], expected)
            # Missing annotations must stop the pipeline instead of labeling normal.
            (audio / "unknown.wav").touch()
            with self.assertRaisesRegex(ValueError, "No annotation"):
                gb.prepare_data_pipeline(root, {}, meta)

    def test_event_parser_and_mixtures_require_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "mixture_devtrain_babycry_000.wav"
            path = root / "event_list_devtrain_babycry.csv"
            path.write_text(f"{name}\t1.5\t2.5\tbabycry\n")
            self.assertEqual(gb.load_event_annotations(root)[name], ("devtrain", (1.5, 2.5, "babycry")))
            (root / name).touch()
            with self.assertRaisesRegex(ValueError, "require --metadata-dir"):
                gb.prepare_data_pipeline(root, {})
            path.write_text(f"{name}\t4\t2\tbabycry\n")
            with self.assertRaisesRegex(ValueError, "Invalid event"):
                gb.load_event_annotations(root)

    def test_empty_audio_is_rejected(self):
        with patch.object(gb.librosa, "load", return_value=(np.array([]), 16000)), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(gb.extract_features("empty.wav"))

    def test_all_windows_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            sr = 16000
            audio = np.zeros(sr * 7, dtype=np.float32)
            audio[sr * 3:] = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(sr * 4) / sr)
            sf.write(path, audio, sr)
            specs = gb.extract_features(path)
            self.assertEqual(specs.shape[0], 3)
            self.assertTrue(np.isfinite(specs).all())
            self.assertGreater(float(specs[1].max()), float(specs[0].max()) + 10)
            sf.write(path, audio[:sr * 6], sr)
            self.assertEqual(len(gb.extract_features(path)), 2)
            sf.write(path, audio[:sr], sr)
            self.assertEqual(len(gb.extract_features(path)), 1)

    def test_source_files_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            ids = {}
            for label in ("normal", "glass_breaking"):
                folder = Path(directory) / label
                folder.mkdir()
                for i in range(10):
                    path = folder / f"{i}.wav"
                    path.touch()
                    ids[str(path)] = len(ids)

            def fake_features(path):
                return np.full((2, 64, 94), ids[path], dtype=np.float32)

            with patch.object(gb, "extract_features", side_effect=fake_features):
                splits = gb.prepare_data_pipeline(directory, {"normal": 0, "glass_breaking": 1})
            groups = [set(x[:, 0, 0]) for x, _ in splits]
            self.assertEqual(sum(len(g) for g in groups), 20)
            for i in range(3):
                self.assertEqual(set(splits[i][1]), {0, 1})
                for j in range(i):
                    self.assertTrue(groups[i].isdisjoint(groups[j]))

    def test_training_learns_and_restores_best_weights(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        x = np.zeros((16, 64, 94), dtype=np.float32)
        x[8:, :32] = 1
        x[8:, 32:] = -1
        y = np.array([0] * 8 + [1] * 8)
        loader = DataLoader(gb.AudioDataset(x, y), batch_size=16)
        model = gb.BaselineAudioCNN(64, 94)
        with contextlib.redirect_stdout(io.StringIO()):
            gb.train_and_evaluate_model(model, loader, loader, epochs=20, lr=0.001)
        self.assertFalse(model.training)
        with torch.no_grad():
            predicted = (model.cpu()(loader.dataset.features) >= 0).int().flatten().numpy()
        np.testing.assert_array_equal(predicted, y)

    def test_prediction_checkpoint_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            model = gb.BaselineAudioCNN(64, 94)
            torch.save({"model_state": model.state_dict(), "input_shape": [64, 94],
                        "mean": 8.0, "std": 2.0}, checkpoint)
            specs = np.full((3, 64, 94), 10.0, dtype=np.float32)
            model.eval()
            with torch.no_grad():
                expected = torch.sigmoid(model(torch.ones(3, 1, 64, 94))).flatten().tolist()
            with patch.object(gb, "extract_features", return_value=specs), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                scores = gb.predict_file("clip.wav", checkpoint)
            self.assertEqual(len(scores), 3)
            np.testing.assert_allclose(scores, expected, rtol=1e-5)
            self.assertIn("6.0-9.0s", output.getvalue())
            torch.save(model.state_dict(), checkpoint)
            with self.assertRaisesRegex(ValueError, "Retrain"):
                gb.predict_file("clip.wav", checkpoint)


if __name__ == "__main__":
    unittest.main()
