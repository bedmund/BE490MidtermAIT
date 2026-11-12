This is the audio CNN prototype for the Midterm Progress report. Google Gemini helped generate the original code.

Install Python 3.10+ and the dependencies:

```sh
pip install -r requirements.txt
```

Place labeled recordings in `dataset/normal/` and `dataset/glass_breaking/`.
Use at least 7 independent files per class (preferably many more, covering varied
backgrounds and recording conditions). Keep duplicate recordings and recordings
from the same session in one split; the automatic split separates files, not sessions.
Every three-second window inherits its file's label. Glass training clips should
therefore contain glass breaking in each window: trim/label event excerpts first
instead of labeling long stretches of ordinary background as glass. Long mixed
recordings can be used for prediction without labels.

Train and evaluate on held-out audio:

```sh
python glassbreak.py --data-dir dataset --epochs 30
```

The original CNN architecture is retained. The pipeline splits source files before
windowing, normalizes using training statistics, weights the loss for class imbalance,
and restores the weights with the lowest validation loss. It prints held-out precision,
recall, F1, and a confusion matrix so that predicting only one class is visible.
These changes do not guarantee real-world accuracy; assess results on representative,
independently labeled recordings.

Classify every non-overlapping three-second window of a test clip:

```sh
python glassbreak.py --predict path/to/test.wav
```

The final partial window is zero-padded; its printed interval includes this padding.
Each interval prints a label and sigmoid glass score (not a calibrated confidence).
WAV, MP3, and FLAC are supported. Empty or unreadable training audio stops training.

Training saves weights and normalization statistics to `best_glass_break_model.pth`.
The checked-in legacy checkpoint is incompatible with the new preprocessing and
must be retrained. Use `--checkpoint path/to/new_model.pth` on both commands to save
and load an alternative checkpoint without overwriting the original.

No live microphone processing or phone notifications are implemented.

Run the regression tests with `python -m unittest discover -s tests`.
