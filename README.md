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
python glassbreak.py --data-dir dataset --metadata-dir metadata --recipes-dir metadata --event-crops 3 --epochs 30 --checkpoint prototype_grouped.pth
```

Each non-overlapping three-second window is positive if it overlaps an annotated
`glassbreak` event at all; all other windows are negative, including babycry and
gunshot events. A filename-only annotation explicitly means no event. An event
ending exactly at a window boundary does not label the following window positive.

The provided `devtest` recordings are reserved for testing. For source-independent
validation, use `--recipes-dir metadata`. The six official
`mixture_recipes_devtrain_*.yaml` and `mixture_recipes_devtest_*.yaml` files are
included beside the CSVs. They were extracted from the Zenodo development
mixture archives 5 and 9 and checked against the current CSVs. These are mixture
recipes, not the source event annotation YAMLs. CSVs alone cannot identify reused
sources. Dataset source: https://zenodo.org/records/401395.

With `--recipes-dir`, recordings sharing an original event file OR background
recording session are joined into connected groups. Adjacent background slices
such as `a001_0_30.wav` and `a001_30_60.wav` are grouped together. The program
searches reproducibly for a group split near 80%/20% of devtrain, with both glass
and non-glass recordings on each side. Counts can differ from a file split.
Sources are also checked for overlap between devtrain and devtest. Missing,
conflicting recipes stop the run: there is no silent file-split fallback when
grouping is requested. If connected groups cannot form a usable split, a seeded
source holdout reserves event/background sources for validation, retains only
training recordings sharing none of those validation sources, and excludes
cross-linked mixtures from this run. Exclusions are printed and recorded as
`excluded` in `split_manifest.csv`; audio files are never deleted, moved, or
relabeled. Candidate source partitions use only devtrain source identities and
labels, never model predictions or test performance. The search prefers a near
80/20 split among retained recordings, representative class proportions, and
fewer exclusions. If no independent split with both classes exists, the run
fails and requires additional independent sources.

Without `--recipes-dir`, the legacy stratified file split is still available and
prints a warning; this does not guarantee independent underlying sources.
`split_manifest.csv` records the assignments and source-group identifiers. The
checkpoint stores recording membership so later evaluation rejects changed splits.
Audio windows/crops from one file never cross splits.

Minimum validation coverage
---------------------------

Source-grouped runs no longer accept a split just because both classes occur.
Before feature extraction or training, BOTH training and validation must contain:

- At least 20 glass-containing recordings.
- At least 10 distinct original annotated glass events.
- At least 5 distinct original glass source recordings.
- At least 5 distinct background recording sessions.
- At least 20 non-glass recordings.

Repeated mixtures of the same original event segment count once toward distinct
events; three-second fragments and added training crops do not increase this
coverage. Different segments from the same source recording count as separate
events but only one independent source. Missing original segment bounds are
counted conservatively as one event per source file.

The seeded search considers several validation fractions (20%-40%) and the
source-holdout strategy, always enforcing source isolation and the coverage
floors. Coverage takes priority over matching an exact 80/20 proportion. No model
scores or test-set results choose the split. Minimums apply to training too so
validation cannot consume nearly all the independent positive training data.
These are practical minimum coverage guards, not statistical confidence guarantees.

Available/accepted counts are printed and written to `split_coverage.json`, even
when coverage fails and training cannot start. Successful runs also include this
information in `summary.json` and the checkpoint. Insufficient coverage or no
qualifying split is an error, not a fallback to tiny validation. Add independent
devtrain material if needed; additional remixes of an existing event/source do
not solve a source-coverage shortage. No audio is deleted or modified.

Retrain after this split change. An old checkpoint evaluated on the new validation
recordings could have seen them during training, so its saved split manifest is
checked and mismatches are rejected. The official devtest collection stays fixed.
These coverage checks require `--recipes-dir`; the legacy file-split mode does
not establish independent source coverage.

Event-focused training crops
----------------------------

For annotation-based training, `--event-crops` defaults to 3. After splitting,
up to three extra three-second crops are added per glass-event training recording,
at different feasible positions around its annotated event. Events no longer
than three seconds fit within these crops; longer events are covered in portions.
Short clips are padded and duplicate crop positions are removed. Every original
window from retained training recordings remains, including difficult background examples and glass
fragments. Nothing is relabeled as normal simply because its overlap is short.
No validation/test crops are added, and file/live inference still scans successive
non-overlapping three-second windows. Set `--event-crops 0` for a controlled
no-augmentation comparison. Retrain to benefit from these crops; evaluating old
weights cannot teach the model new features.

The original CNN architecture is retained. The pipeline normalizes using training
statistics, weights the loss for class imbalance,
and restores the weights with the lowest validation loss. It prints held-out precision,
recall, F1, and a confusion matrix so that predicting only one class is visible.
These changes do not guarantee real-world accuracy; assess results on representative,
independently labeled recordings.
It reports per-split window counts and balanced validation accuracy, because
correct labels produce many more background windows than glass windows. Review
glass recall and normal recall as well as ordinary accuracy.

Validation thresholds and error reports
--------------------------------------

Each training epoch also prints validation glass precision, recall, and F1 at the
fixed 0.5 threshold. Model weights are still selected by lowest validation loss.
After restoring those weights, the program evaluates every distinct validation
score as a candidate threshold, plus 0, 0.5, and 1. It selects maximum validation
glass F1; ties prefer higher precision, then closeness to 0.5, then the higher
threshold. Predictions use `score >= threshold`. Test data never selects the
threshold. The selected threshold is saved alongside the model and normalization.

You can evaluate an existing annotation-trained model without retraining:

```sh
python glassbreak.py --evaluate --data-dir dataset --metadata-dir metadata --recipes-dir metadata --checkpoint prototype_grouped.pth --report-dir reports_grouped
```

Use the same audio collection and metadata as when that checkpoint was trained,
so deterministic recording splits reproduce its held-out validation set. This
does not recover historical splits if recordings were added, removed, or renamed.
Use the same `--recipes-dir` setting as training. A legacy checkpoint without a
split manifest cannot be evaluated on a newly grouped validation split: retrain
first, since the old model may have seen those recordings. `--evaluate` never adds
training crops or changes weights. The saved normalization is reused. A new
`prototype_grouped_calibrated.pth` is
written by default; `--calibrated-checkpoint path/to/output.pth` chooses another
destination. Here "calibrated" means threshold selection, not probability
calibration. The existing checkpoint is not overwritten by default.

Training/evaluation writes the following into `--report-dir` (default `reports/`):

- `validation_thresholds.csv`: precision, recall, F1, accuracy, balanced accuracy,
  and confusion counts for every candidate threshold.
- `validation_predictions.csv` and `test_predictions.csv`: every window's audio
  path, filename, start/end seconds, event annotation, actual/predicted label,
  glass score, and selected threshold.
- `validation_errors.csv` and `test_errors.csv`: only false positives and false
  negatives. False positives are sorted by highest score first; false negatives
  by lowest score first, to help prioritize listening.
- `summary.json`: selected threshold, selection rule, and validation/test metrics
  at both the selected threshold and the old 0.5 baseline.
- `split_manifest.csv`: recording membership and source groups (when enabled).
- `validation_events.csv` and `test_events.csv`: timed glass-event detection or
  miss per recording and the number of false-alarm episodes.
- `validation_alarm_episodes.csv` and `test_alarm_episodes.csv`: start/end times of
  merged adjacent positive windows and whether each episode matched a glass event.

Event summaries merge adjacent positive windows within each file into alarm
episodes. An episode matches a glass event if their intervals overlap at all;
at most one episode matches each event. Unmatched episodes count as false positives
and unmatched events as false negatives. Thus missing one fragment need not mean
missing the whole event. These are coarse overlap-based diagnostics, NOT official
DCASE onset-collar metrics. Threshold selection remains based on validation WINDOW
F1, not event F1. Event reports require timed annotations and are omitted for
folder-only datasets. Retain window-level metrics too: a long positive episode can
overlap glass while also covering substantial background. This is not a substitute
for measuring real alarm timing/false alarms.

Reports are window-level, not event-level alarm counts. Final partial windows
include padding, so their nominal end time may exceed the real clip duration.
The report directory is reused/overwritten on each run; choose a different one
to retain earlier experiments. Reports do not copy or modify audio files.
Use validation errors to develop improvements and reserve test reports for final
assessment; inspecting and tuning against test errors compromises a held-out test.
Threshold evaluation does not change weights or guarantee a higher test F1.
Event-focused crops affect training; source independence requires recipe grouping.

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

Listen to a microphone with an existing trained checkpoint (no CSVs or dataset needed):

```sh
python glassbreak.py --list-input-devices
python glassbreak.py --live --checkpoint prototype_annotated.pth
```

To select a microphone by its device index from the list:

```sh
python glassbreak.py --live --device 2 --checkpoint prototype_annotated.pth
```

On Ubuntu/Linux, PortAudio may need to be installed before microphone capture:

```sh
sudo apt install libportaudio2
python -m pip install -r requirements.txt
```

If running inside a VM, enable microphone/audio-input passthrough in the VM and
allow microphone access on the host. The device list must show an input device
available inside the VM; an SSH session does not automatically forward your local
microphone. Device indices may change, so check the list after reconnecting.

Live capture uses mono audio at the device's default sample rate, resamples to
16 kHz, and applies exactly the same three-second log-mel features and checkpoint
normalization as file prediction. A callback captures successive windows while
the main thread runs inference. Results show elapsed capture intervals and glass
scores after each completed window (at least three seconds plus processing/device
latency). Ctrl+C closes the microphone; unfinished audio is discarded. Audio is
held in memory only and is not saved or transmitted. Capture errors or a full
four-window queue stop monitoring with an error rather than silently losing audio.

Prediction uses the checkpoint's saved threshold by default; older checkpoints
without one fall back to 0.5. Both `--live` and `--predict` accept an explicit
`--threshold`, e.g. `--threshold 0.7`, which overrides the saved value without
modifying the checkpoint. Training/evaluation selects thresholds using validation
data and rejects this manual override. To use a separately evaluated checkpoint,
pass `--checkpoint prototype_annotated_calibrated.pth` for file or live prediction.
Live capture does not improve the existing model's detection accuracy; the
previously observed false alarms and missed events remain relevant.

Gmail email alerts
------------------

Email alerts are disabled by default. Copy `email_config.example.py` to
`email_config.py` beside the script, then edit and uncomment all three dummy fields:
`GMAIL_SENDER`, `EMAIL_RECIPIENT`, and `GMAIL_APP_PASSWORD`. Use your dedicated Gmail
account as sender and your iCloud (or other) address as recipient. Use a Google App
Password, not your ordinary password. App Password availability depends on your
Google account security settings and requires 2-Step Verification. No recipient
password is needed. `email_config.py` is Git-ignored; never commit real credentials.

```sh
python glassbreak.py --live --checkpoint prototype_annotated.pth --email-alerts
```

Combine email and the simulated alarm if desired:

```sh
python glassbreak.py --live --checkpoint prototype_annotated.pth --alarm simulate --email-alerts --email-cooldown 60
```

Each eligible live glass detection queues a plain-text email containing the UTC
detection time, elapsed audio interval, and glass score. No audio is attached.
Sending uses Gmail SMTP over verified TLS on port 465 with a 10-second socket
timeout. A background worker keeps network requests off the audio processing
thread. There is at most one queued message, and the default cooldown is 60
seconds between queued alerts (including attempts that later fail). This limits
false-alarm email volume; it does not guarantee delivery or detector accuracy.

SMTP/network failures are logged by exception type without printing credentials;
live detection continues. There are no automatic retries, avoiding duplicate
messages after ambiguous SMTP failures. Later eligible detections may send new
alerts. On shutdown, pending messages are discarded; an in-flight send may finish.
Gmail acceptance is not confirmation of delivery to your inbox. Delivery and phone
notifications depend on mail-provider and iPhone settings. Email is not P2P.

The implementation uses Python's standard library, so no extra requirements.txt
package is needed. Email tests mock SMTP and do not send messages. Configure and
test your own account before a demonstration.

Optional light and active-buzzer alarm
------------------------------------

For the project without Raspberry Pi hardware, simulate both outputs:

```sh
python glassbreak.py --live --checkpoint prototype_annotated.pth --alarm simulate --alarm-seconds 5
```

The console displays `ALARM (simulate): light ON, buzzer ON` on a detection and
an OFF message after the hold interval. Simulation does not access GPIO, emit
sound from the computer speaker, or switch a physical lamp. Ordinary `--live`
leaves the alarm disabled.

For a future Raspberry Pi with GPIO-connected components:

```sh
python glassbreak.py --live --checkpoint prototype_annotated.pth --alarm gpio --led-pin 17 --buzzer-pin 27 --alarm-seconds 5
```

Pin numbers use BCM numbering, not physical header positions. Outputs are
active-high and initialized OFF. Use a GPIO-compatible LED with a series resistor
and an active buzzer or suitable driver circuit; powered lamps and larger buzzers
require a rated driver, not a direct GPIO connection. This interface controls GPIO
outputs; arbitrary USB lights/buzzers need their own controller integration.

`gpiozero` is included in requirements.txt. Actual GPIO use additionally needs a
compatible Pi pin backend and GPIO permissions; Pi 5 requires lgpio. See the
[GPIO Zero pin backend documentation](https://gpiozero.readthedocs.io/en/stable/api_pins.html).
Hardware mode reports an initialization error on systems without GPIO; it does
not silently fall back to simulation.

An alarm triggers only for captured windows whose score meets `--threshold`.
Model warmup, training, and file prediction do not activate it. Both outputs stay
on for `--alarm-seconds` after the most recent detection (default 3 seconds).
Normal windows do not end the hold early. A background timer handles expiry,
so the alarm does not pause microphone processing. Ctrl+C or a capture/inference
failure cancels the timer, turns outputs OFF, and releases the GPIO devices.
GPIO errors during timer expiry are surfaced to the monitoring loop. Normal
process cleanup cannot cover forced termination or power loss.

The alarm uses the existing model and threshold; false glass detections also
trigger it. Email is sent only when separately enabled with `--email-alerts`. GPIO behavior is tested
with simulated devices; deployment with a physical Pi/light/buzzer is unverified.

Run the regression tests with `python -m unittest discover -s tests`.
