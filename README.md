This is the audio CNN prototype for the Midterm Progress report. Google Gemini helped generate the original code.

Install Python 3.10+ and the dependencies:

```sh
pip install -r requirements.txt
```

For TUT Rare Sound Events 2017 mixtures, keep your existing audio under
`dataset/normal/` and `dataset/glass_breaking/` (nested original audio directories
also work). Preserve original filenames. Folder names do not determine the labels:
the event-list annotations do. You can use a subset of the audio, but every audio
file you include must have an annotation. Duplicate audio filenames are rejected.

Place the six current `event_list_devtrain_*.csv` and `event_list_devtest_*.csv`
files in `metadata/` (or point `--metadata-dir` at their extracted directory).
Do not use `*.csv_old_dontuse` files or obsolete audio. The current babycry lists
exclude some obsolete recordings, so unmatched audio must be removed from the
training input. Missing metadata is an error, not an assumed negative label.

Train and evaluate on held-out audio:

```sh
python glassbreak.py --data-dir dataset --metadata-dir metadata --epochs 30 --checkpoint prototype_annotated.pth
```

Each non-overlapping three-second window is positive if it overlaps an annotated
`glassbreak` event at all; all other windows are negative, including babycry and
gunshot events. A filename-only annotation explicitly means no event. An event
ending exactly at a window boundary does not label the following window positive.

The provided `devtest` recordings are reserved for testing. Validation uses 20% of
the available `devtrain` recordings, stratified by whether they contain glass.
Mixtures can reuse source sounds and backgrounds, so this file-level validation
split may share underlying sources with training; the official devtest split is
the independent final assessment. Source-grouped validation would additionally
require the mixture recipes. Audio windows from one file never cross splits.

The original CNN architecture is retained. The pipeline normalizes using training
statistics, weights the loss for class imbalance,
and restores the weights with the lowest validation loss. It prints held-out precision,
recall, F1, and a confusion matrix so that predicting only one class is visible.
These changes do not guarantee real-world accuracy; assess results on representative,
independently labeled recordings.
It reports per-split window counts and balanced validation accuracy, because
correct labels produce many more background windows than glass windows. Review
glass recall and normal recall as well as ordinary accuracy.

For other, manually curated datasets, the original folder-label mode remains
available by omitting `--metadata-dir`. This mode requires at least 7 independent
recordings per class and glass breaking in every window of glass-labeled files.
TUT mixture filenames are rejected in that mode to prevent accidental mislabeling.

Classify every non-overlapping three-second window of a test clip:

```sh
python glassbreak.py --predict path/to/test.wav --checkpoint prototype_annotated.pth
```

The final partial window is zero-padded; its printed interval includes this padding.
Each interval prints a label and sigmoid glass score (not a calibrated confidence).
WAV, MP3, and FLAC are supported. Empty or unreadable training audio stops training.

Training saves weights and normalization statistics to `best_glass_break_model.pth`.
The checked-in legacy checkpoint is incompatible with the new preprocessing and
must be retrained. Use `--checkpoint path/to/new_model.pth` on both commands to save
and load an alternative checkpoint without overwriting the original.
Models previously trained with folder-level mixture labels should also be retrained.

No live microphone processing or phone notifications are implemented.

Run the regression tests with `python -m unittest discover -s tests`.
