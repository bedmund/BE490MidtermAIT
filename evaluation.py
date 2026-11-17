"""Window-level evaluation and validation-only threshold selection."""

import csv
import json
from pathlib import Path
import numpy as np


def metrics_from_counts(tn, fp, fn, tp):
    tn, fp, fn, tp = map(int, (tn, fp, fn, tp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return dict(precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                accuracy=(tp + tn) / (tn + fp + fn + tp) if tn + fp + fn + tp else 0.0,
                balanced_accuracy=(recall + specificity) / 2,
                true_negative=tn, false_positive=fp, false_negative=fn, true_positive=tp)


def evaluate_thresholds(labels, scores, thresholds=None):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Expected matching nonempty one-dimensional labels and scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected binary labels and finite scores between 0 and 1")
    # Test every distinct decision boundary, plus 0.5 and the endpoints.
    thresholds = np.unique(np.r_[scores, 0.0, 0.5, 1.0]) if thresholds is None else np.asarray(thresholds)
    if thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Thresholds must be between 0 and 1")
    order = np.argsort(scores)
    sorted_scores = scores[order]
    positive_prefix = np.r_[0, np.cumsum(labels[order])]
    rows = []
    for threshold in thresholds:
        below = np.searchsorted(sorted_scores, threshold, side="left")
        fn = positive_prefix[below]
        tn = below - fn
        tp = positive_prefix[-1] - fn
        fp = len(labels) - below - tp
        rows.append(dict(threshold=float(threshold), **metrics_from_counts(tn, fp, fn, tp)))
    return rows


def select_validation_threshold(labels, scores):
    if set(np.asarray(labels).tolist()) != {0, 1}:
        raise ValueError("Threshold selection requires both validation classes")
    rows = evaluate_thresholds(labels, scores)
    # Tie break: higher precision, closest to 0.5, then higher threshold.
    selected = max(rows, key=lambda row: (row['f1'], row['precision'], -abs(row['threshold'] - 0.5), row['threshold']))
    return selected, rows


def write_csv(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_window_reports(directory, split, references, labels, scores, threshold):
    if not (len(references) == len(labels) == len(scores)):
        raise ValueError("Report references, labels, and scores must align")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for reference, label, score in zip(references, labels, scores):
        predicted = int(score >= threshold)
        outcome = ('false_positive' if predicted else 'false_negative') if predicted != label else ''
        rows.append(dict(reference, split=split, actual_label='glass_breaking' if label else 'normal',
                         predicted_label='glass_breaking' if predicted else 'normal',
                         glass_score=float(score), threshold=float(threshold), error_type=outcome))
    fields = ['split', 'file_path', 'filename', 'start_seconds', 'end_seconds',
              'event_label', 'event_start_seconds', 'event_end_seconds',
              'actual_label', 'predicted_label', 'glass_score', 'threshold', 'error_type']
    write_csv(directory / f'{split}_predictions.csv', rows, fields)
    errors = sorted((row for row in rows if row['error_type']),
                    key=lambda row: (0 if row['error_type'] == 'false_positive' else 1,
                                     -row['glass_score'] if row['error_type'] == 'false_positive' else row['glass_score']))
    write_csv(directory / f'{split}_errors.csv', errors, fields)


def evaluate_and_report(validation, test, directory):
    """Each split is (labels, scores, references); test never selects a threshold."""
    val_labels, val_scores, val_references = validation
    selected, rows = select_validation_threshold(val_labels, val_scores)
    threshold = selected['threshold']
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / 'validation_thresholds.csv', rows, list(rows[0]))
    summary = dict(metric_unit='three-second windows', selection='maximum validation glass F1',
                   selected_threshold=threshold, validation=selected,
                   validation_at_0_5=evaluate_thresholds(val_labels, val_scores, [0.5])[0])
    for split, (labels, scores, references) in [('validation', validation), ('test', test)]:
        write_window_reports(directory, split, references, labels, scores, threshold)
        if split == 'test':
            summary['test'] = evaluate_thresholds(labels, scores, [threshold])[0]
            summary['test_at_0_5'] = evaluate_thresholds(labels, scores, [0.5])[0]
    with (directory / 'summary.json').open('w', encoding='utf-8') as output:
        json.dump(summary, output, indent=2, allow_nan=False)
    return threshold, summary
