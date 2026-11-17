import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import evaluation as ev


class EvaluationTests(unittest.TestCase):
    def references(self, count):
        return [dict(file_path=f'/audio/clip{i}.wav', filename=f'clip{i}.wav',
                     start_seconds=i * 3, end_seconds=(i + 1) * 3,
                     event_label='', event_start_seconds='', event_end_seconds='') for i in range(count)]

    def test_validation_threshold_and_metrics(self):
        selected, rows = ev.select_validation_threshold([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual(selected['threshold'], 0.35)
        self.assertEqual(selected['precision'], 2 / 3)
        self.assertEqual(selected['recall'], 1)
        self.assertEqual(selected['f1'], 0.8)
        baseline = next(row for row in rows if row['threshold'] == 0.5)
        self.assertEqual(baseline['false_negative'], 1)
        self.assertEqual(baseline['f1'], 2 / 3)

    def test_equal_scores_boundary_and_zero_division(self):
        rows = ev.evaluate_thresholds([0, 1], [0.5, 0.5], [0.5, 0.6])
        self.assertEqual(rows[0]['true_positive'], 1)
        self.assertEqual(rows[0]['false_positive'], 1)
        self.assertEqual(rows[1]['precision'], 0)
        self.assertEqual(rows[1]['recall'], 0)
        self.assertEqual(rows[1]['f1'], 0)

    def test_reports_and_test_labels_cannot_change_selected_threshold(self):
        references = self.references(4)
        validation = ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], references)
        with tempfile.TemporaryDirectory() as directory:
            threshold, _ = ev.evaluate_and_report(validation, ([0, 1, 0, 1], [0.9, 0.1, 0.2, 0.8], references), directory)
            with (Path(directory) / 'test_errors.csv').open(newline='') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row['error_type'] for row in rows], ['false_positive', 'false_negative'])
            self.assertEqual(rows[0]['filename'], 'clip0.wav')
            self.assertEqual(rows[1]['start_seconds'], '3')
            self.assertEqual(rows[1]['end_seconds'], '6')
            changed, _ = ev.evaluate_and_report(validation, ([1, 0, 1, 0], [0.9, 0.1, 0.2, 0.8], references), directory)
            self.assertEqual(changed, threshold)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['selected_threshold'], threshold)
            self.assertIn('test_at_0_5', summary)

    def test_empty_error_report_still_has_header(self):
        with tempfile.TemporaryDirectory() as directory:
            ev.write_window_reports(directory, 'test', self.references(2), [0, 1], [0.1, 0.9], 0.5)
            with (Path(directory) / 'test_errors.csv').open(newline='') as source:
                reader = csv.DictReader(source)
                self.assertIn('filename', reader.fieldnames)
                self.assertEqual(list(reader), [])

    def test_bad_scores_and_single_class_validation_are_rejected(self):
        for labels, scores in [([], []), ([0, 1], [float('nan'), 0.5]), ([0, 1], [1.1, 0.5])]:
            with self.assertRaises(ValueError):
                ev.evaluate_thresholds(labels, scores)
        with self.assertRaises(ValueError):
            ev.select_validation_threshold([0, 0], [0.1, 0.2])
