import os
import argparse
import copy
import csv
import queue
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        # Reshape to (Batch, Channels, Height, Width) -> (B, 1, n_mels, time_frames)
        self.features = self.features.unsqueeze(1) 
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def extract_audio_features(audio, sr=16000, duration=3.0, n_mels=64):
    if len(audio) == 0:
        raise ValueError("Audio clip is empty")
    target_length = int(sr * duration)
    if target_length <= 0:
        raise ValueError("Window duration must be positive")
    features = []
    for start in range(0, len(audio), target_length):
        segment = audio[start:start + target_length]
        segment = np.pad(segment, (0, target_length - len(segment)))
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
        # Fixed reference preserves level differences between quiet and loud windows.
        log_mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
        features.append(log_mel_spec)
    return np.array(features)


def extract_features(file_path, sr=16000, duration=3.0, n_mels=64):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
        return extract_audio_features(audio, sr, duration, n_mels)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_event_annotations(metadata_dir):
    annotations = {}
    for path in sorted(Path(metadata_dir).rglob("event_list_*.csv")):
        parts = path.stem.split("_")
        if len(parts) != 4 or parts[2] not in ("devtrain", "devtest"):
            continue
        subset, event_class = parts[2:]
        with path.open(encoding="utf-8-sig", newline="") as source:
            for line_number, row in enumerate(csv.reader(source, delimiter="\t"), 1):
                if not row:
                    continue
                if len(row) not in (1, 4):
                    raise ValueError(f"Invalid annotation: {path}:{line_number}")
                name = row[0].replace("\\", "/").rsplit("/", 1)[-1]
                if not name.startswith(f"mixture_{subset}_{event_class}_"):
                    raise ValueError(f"Annotation filename does not match its list: {name}")
                event = None
                if len(row) == 4:
                    start, end = float(row[1]), float(row[2])
                    if not np.isfinite([start, end]).all() or not 0 <= start < end or row[3] != event_class:
                        raise ValueError(f"Invalid event: {path}:{line_number}")
                    event = (start, end, row[3])
                if name in annotations:
                    raise ValueError(f"Duplicate annotation for {name}")
                annotations[name] = (subset, event)
    if not annotations:
        raise ValueError(f"No current event_list_*.csv annotations found in {metadata_dir}")
    return annotations


def prepare_mixture_data(data_dir, metadata_dir):
    annotations = load_event_annotations(metadata_dir)
    recordings = {"devtrain": [], "devtest": []}
    seen = set()
    for path in sorted(Path(data_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".wav", ".mp3", ".flac"):
            continue
        if path.name in seen:
            raise ValueError(f"Duplicate audio filename: {path.name}")
        seen.add(path.name)
        if path.name not in annotations:
            raise ValueError(f"No annotation for {path.name}; missing metadata is not a normal label. "
                             "Exclude obsolete _old_dontuse recordings and preserve original filenames.")
        subset, event = annotations[path.name]
        recordings[subset].append((str(path), event))

    # Official devtest remains untouched. Validation uses only devtrain recordings.
    try:
        train, validation = train_test_split(
            recordings["devtrain"], test_size=0.2, random_state=42,
            stratify=[int(event is not None and event[2] == "glassbreak")
                      for _, event in recordings["devtrain"]])
    except ValueError as error:
        raise ValueError("Need enough annotated devtrain recordings with and without glass for validation.") from error

    def build_windows(records, split):
        features, labels = [], []
        for path, event in records:
            specs = extract_features(path)
            if specs is None:
                raise ValueError(f"Cannot use unreadable audio: {path}")
            if event is not None and event[1] > len(specs) * 3:
                raise ValueError(f"Annotation exceeds windowed audio duration: {path}")
            for index, spec in enumerate(specs):
                # Half-open intervals: an event on a boundary belongs to the next window.
                glass = (event is not None and event[2] == "glassbreak"
                         and event[0] < (index + 1) * 3 and event[1] > index * 3)
                features.append(spec)
                labels.append(int(glass))
        if set(labels) != {0, 1}:
            raise ValueError(f"{split} must contain both glass and non-glass windows")
        print(f"{split}: {len(records)} recordings, {sum(labels)} glass windows, "
              f"{len(labels) - sum(labels)} normal windows")
        return np.array(features), np.array(labels)

    return (build_windows(train, "train"), build_windows(validation, "validation"),
            build_windows(recordings["devtest"], "test"))


def prepare_data_pipeline(data_dir, label_map, metadata_dir=None):
    if metadata_dir is not None:
        return prepare_mixture_data(data_dir, metadata_dir)
    if any(Path(data_dir).rglob("mixture_*.wav")):
        raise ValueError("TUT mixtures require --metadata-dir with the current event_list_*.csv files; "
                         "folder names cannot label their windows correctly.")
    files, labels = [], []
    
    for label_name, label_idx in label_map.items():
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(class_dir):
            continue
        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(class_dir, file_name)
                files.append(file_path)
                labels.append(label_idx)

    # Split recordings BEFORE windowing so related windows cannot leak into testing.
    try:
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            files, labels, test_size=0.3, random_state=42, stratify=labels)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
    except ValueError as e:
        raise ValueError("Provide enough independent recordings in both normal and glass_breaking "
                         "for stratified train/validation/test splits (at least 7 per class).") from e

    def build_windows(paths, file_labels):
        features, window_labels = [], []
        for path, label in zip(paths, file_labels):
            specs = extract_features(path)
            if specs is None:
                raise ValueError(f"Cannot use unreadable audio: {path}")
            features.extend(specs)
            window_labels.extend([label] * len(specs))
        if set(window_labels) != {0, 1}:
            raise ValueError("Each split must contain both normal and glass-breaking recordings")
        return np.array(features), np.array(window_labels)

    return (build_windows(train_files, train_labels),
            build_windows(val_files, val_labels), build_windows(test_files, test_labels))

class BaselineAudioCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(BaselineAudioCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        dummy_input = torch.zeros(1, 1, input_height, input_width)
        flattened_dim = self.conv_block(dummy_input).view(1, -1).size(1)
        
        self.fc_block = nn.Sequential(
            nn.Linear(flattened_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            # Raw logits for BCEWithLogitsLoss; apply sigmoid during prediction.
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

def train_and_evaluate_model(model, train_loader, val_loader, epochs=10, lr=0.0001):
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    labels = train_loader.dataset.labels
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    if not positives or not negatives:
        raise ValueError("Training requires both classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        model.eval()
        correct, total = 0, 0
        class_correct, class_total = [0, 0], [0, 0]
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for class_id in (0, 1):
                    mask = labels == class_id
                    class_total[class_id] += mask.sum().item()
                    class_correct[class_id] += ((predicted == labels) & mask).sum().item()
                
        val_acc = correct / total if total > 0 else 0.0
        val_loss /= total
        balanced_acc = sum(c / n if n else 0 for c, n in zip(class_correct, class_total)) / 2
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader.dataset):.4f} "
              f"| Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} "
              f"| Val Balanced Accuracy: {balanced_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    model.eval()
    print(f"Training Complete. Validation Accuracy at best loss: {best_val_acc:.4f}")
    return model


def load_prediction_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state" not in checkpoint:
        raise ValueError("This checkpoint uses the old preprocessing. Retrain with this script first.")
    if not np.isfinite(checkpoint["mean"]) or not np.isfinite(checkpoint["std"]) or checkpoint["std"] <= 0:
        raise ValueError("Checkpoint normalization statistics are invalid")
    model = BaselineAudioCNN(*checkpoint["input_shape"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def predict_features(model, checkpoint, specs):
    inputs = torch.tensor((specs - checkpoint["mean"]) / checkpoint["std"], dtype=torch.float32).unsqueeze(1)
    probabilities = []
    with torch.no_grad():
        for batch in inputs.split(16):
            probabilities.extend(torch.sigmoid(model(batch)).flatten().tolist())
    return probabilities


def print_prediction(index, probability, threshold=0.5):
    label = "glass_breaking" if probability >= threshold else "normal"
    print(f"{index * 3:.1f}-{(index + 1) * 3:.1f}s: {label} (glass score: {probability:.3f})", flush=True)


def predict_file(file_path, checkpoint_path="best_glass_break_model.pth", threshold=0.5):
    model, checkpoint = load_prediction_model(checkpoint_path)
    specs = extract_features(file_path)
    if specs is None:
        raise ValueError(f"Cannot read audio: {file_path}")
    probabilities = predict_features(model, checkpoint, specs)
    for index, probability in enumerate(probabilities):
        print_prediction(index, probability, threshold)
    return probabilities


def get_audio_backend():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        raise RuntimeError("Microphone capture needs sounddevice and PortAudio. "
                           "Install requirements.txt; on Ubuntu install libportaudio2.") from error
    return sd


def predict_live(checkpoint_path="best_glass_break_model.pth", device=None, threshold=0.5,
                 alarm_mode="off", led_pin=17, buzzer_pin=27, alarm_seconds=3.0):
    model, checkpoint = load_prediction_model(checkpoint_path)
    sd = get_audio_backend()
    # Capture at the microphone's native rate, then resample to the training rate.
    input_device = sd.query_devices(device, "input")
    sample_rate = int(round(input_device["default_samplerate"]))
    sd.check_input_settings(device=device, channels=1, dtype="float32", samplerate=sample_rate)
    # Load feature kernels/model inference before recording so cold-start work
    # cannot fill the capture queue before the first real prediction.
    print("Preparing live audio processing...", flush=True)
    if sample_rate != 16000:
        librosa.resample(np.zeros(sample_rate * 3, dtype=np.float32), orig_sr=sample_rate, target_sr=16000)
    predict_features(model, checkpoint, extract_audio_features(np.zeros(48000, dtype=np.float32)))
    windows = queue.Queue(maxsize=4)
    errors = queue.SimpleQueue()

    def capture(indata, frames, time_info, status):
        # Never run the CNN or block on the audio callback thread.
        if status:
            errors.put(f"Microphone capture interrupted: {status}")
            raise sd.CallbackAbort
        if frames != sample_rate * 3:
            errors.put("Microphone returned an incomplete three-second window")
            raise sd.CallbackAbort
        try:
            windows.put_nowait(indata[:, 0].copy())
        except queue.Full:
            errors.put("Inference is falling behind microphone capture; stopped to avoid losing audio silently")
            raise sd.CallbackAbort

    try:
        from alarm import AlarmOutputs
        alarm_context = (nullcontext(None) if alarm_mode == "off" else
                         AlarmOutputs(alarm_mode, led_pin, buzzer_pin, alarm_seconds))
        with alarm_context as alarm, sd.InputStream(device=device, samplerate=sample_rate, channels=1, dtype="float32",
                            blocksize=sample_rate * 3, callback=capture) as stream:
            print(f"Listening on {input_device['name']} in 3-second windows. Press Ctrl+C to stop.", flush=True)
            index = 0
            while True:
                if alarm is not None:
                    alarm.check()
                if not errors.empty():
                    raise RuntimeError(errors.get())
                try:
                    audio = windows.get(timeout=0.2)
                except queue.Empty:
                    if not stream.active:
                        raise RuntimeError("Microphone stream stopped unexpectedly")
                    continue
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                specs = extract_audio_features(audio)
                probability = predict_features(model, checkpoint, specs)[0]
                print_prediction(index, probability, threshold)
                if alarm is not None and probability >= threshold:
                    alarm.trigger()
                index += 1
    except KeyboardInterrupt:
        print("\nMicrophone capture stopped. Any unfinished window was discarded.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on labeled recordings or classify every 3-second window.")
    parser.add_argument("--data-dir", default="./dataset")
    parser.add_argument("--metadata-dir", help="Directory containing current TUT mixture event-list CSVs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predict", metavar="AUDIO_FILE")
    mode.add_argument("--live", action="store_true", help="Classify microphone audio until Ctrl+C")
    mode.add_argument("--list-input-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, help="Microphone device index (default: system input)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Glass score threshold for prediction")
    parser.add_argument("--alarm", choices=("off", "simulate", "gpio"), default="off",
                        help="Live light/buzzer alarm (simulate needs no hardware)")
    parser.add_argument("--led-pin", type=int, default=17, help="LED BCM GPIO pin")
    parser.add_argument("--buzzer-pin", type=int, default=27, help="Active buzzer BCM GPIO pin")
    parser.add_argument("--alarm-seconds", type=float, default=3.0, help="Alarm hold time after each detection")
    parser.add_argument("--checkpoint", default="best_glass_break_model.pth")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if args.alarm != "off":
        if not args.live:
            parser.error("--alarm requires --live")
        if not np.isfinite(args.alarm_seconds) or args.alarm_seconds <= 0:
            parser.error("--alarm-seconds must be positive and finite")
        if args.led_pin == args.buzzer_pin or not all(0 <= pin <= 27 for pin in (args.led_pin, args.buzzer_pin)):
            parser.error("Use distinct --led-pin and --buzzer-pin BCM numbers between 0 and 27")
    torch.manual_seed(42)
    np.random.seed(42)
    DATA_DIR = args.data_dir
    LABEL_MAP = {"normal": 0, "glass_breaking": 1}
    
    if args.list_input_devices:
        print(get_audio_backend().query_devices())
    elif args.live:
        predict_live(args.checkpoint, args.device, args.threshold, args.alarm,
                     args.led_pin, args.buzzer_pin, args.alarm_seconds)
    elif args.predict:
        predict_file(args.predict, args.checkpoint, args.threshold)
    elif os.path.exists(DATA_DIR):
        train_data, val_data, test_data = prepare_data_pipeline(DATA_DIR, LABEL_MAP, args.metadata_dir)
        # Fit normalization only on training audio; reuse it for all later inputs.
        mean = float(train_data[0].mean())
        std = max(float(train_data[0].std()), 1e-6)
        train_data = ((train_data[0] - mean) / std, train_data[1])
        val_data = ((val_data[0] - mean) / std, val_data[1])
        test_data = ((test_data[0] - mean) / std, test_data[1])
        
        train_loader = DataLoader(AudioDataset(*train_data), batch_size=16, shuffle=True)
        val_loader = DataLoader(AudioDataset(*val_data), batch_size=16, shuffle=False)
        
        sample_spec = train_data[0][0]
        model = BaselineAudioCNN(input_height=sample_spec.shape[0], input_width=sample_spec.shape[1])
        
        trained_model = train_and_evaluate_model(model, train_loader, val_loader, epochs=args.epochs)
        torch.save({"model_state": trained_model.cpu().state_dict(),
                    "input_shape": list(sample_spec.shape), "mean": mean, "std": std}, args.checkpoint)
        test_loader = DataLoader(AudioDataset(*test_data), batch_size=16, shuffle=False)
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                predictions.extend((torch.sigmoid(trained_model(inputs)) >= 0.5).int().flatten().tolist())
        print("Test confusion matrix (rows: actual, columns: predicted; normal, glass_breaking):")
        print(confusion_matrix(test_data[1], predictions, labels=[0, 1]))
        print(classification_report(test_data[1], predictions, labels=[0, 1],
                                    target_names=list(LABEL_MAP), zero_division=0))
    else:
        print(f"Directory '{DATA_DIR}' not found. Please organize your audio subdirectories ('normal' and 'glass_breaking') to run the pipeline.")
