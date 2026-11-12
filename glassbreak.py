import os
import argparse
import copy
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

def extract_features(file_path, sr=16000, duration=3.0, n_mels=64):
    try:
        audio, _ = librosa.load(file_path, sr=sr)
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
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_data_pipeline(data_dir, label_map):
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
                
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader.dataset):.4f} | Val Accuracy: {val_acc:.4f}")
        val_loss /= total
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_state)
    model.eval()
    print(f"Training Complete. Validation Accuracy at best loss: {best_val_acc:.4f}")
    return model


def predict_file(file_path, checkpoint_path="best_glass_break_model.pth"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state" not in checkpoint:
        raise ValueError("This checkpoint uses the old preprocessing. Retrain with this script first.")
    specs = extract_features(file_path)
    if specs is None:
        raise ValueError(f"Cannot read audio: {file_path}")
    model = BaselineAudioCNN(*checkpoint["input_shape"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    inputs = torch.tensor((specs - checkpoint["mean"]) / checkpoint["std"], dtype=torch.float32).unsqueeze(1)
    probabilities = []
    with torch.no_grad():
        for batch in inputs.split(16):
            probabilities.extend(torch.sigmoid(model(batch)).flatten().tolist())
    for i, probability in enumerate(probabilities):
        label = "glass_breaking" if probability >= 0.5 else "normal"
        print(f"{i * 3:.1f}-{(i + 1) * 3:.1f}s: {label} (glass score: {probability:.3f})")
    return probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on labeled recordings or classify every 3-second window.")
    parser.add_argument("--data-dir", default="./dataset")
    parser.add_argument("--predict", metavar="AUDIO_FILE")
    parser.add_argument("--checkpoint", default="best_glass_break_model.pth")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    DATA_DIR = args.data_dir
    LABEL_MAP = {"normal": 0, "glass_breaking": 1}
    
    if args.predict:
        predict_file(args.predict, args.checkpoint)
    elif os.path.exists(DATA_DIR):
        train_data, val_data, test_data = prepare_data_pipeline(DATA_DIR, LABEL_MAP)
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
