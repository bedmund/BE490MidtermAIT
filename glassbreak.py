import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

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
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        target_length = int(sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-6)
        
        return log_mel_spec
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_data_pipeline(data_dir, label_map):
    features, labels = [], []
    
    for label_name, label_idx in label_map.items():
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(class_dir, file_name)
                spec = extract_features(file_path)
                if spec is not None:
                    features.append(spec)
                    labels.append(label_idx)
                    
    X = np.array(features)
    y = np.array(labels)
    
    indicies = np.random.permutation(len(X))
    X = X[indicies]
    y = y[indicies]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

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
            #nn.Sigmoid() # Binary classification: Normal (0) vs Glass Breaking (1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x

def train_and_evaluate_model(model, train_loader, val_loader, epochs=10, lr=0.0001):
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_acc = 0.0
    
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
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                predicted = (probs >= 0.5).float()

                #predicted = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_loader.dataset):.4f} | Val Accuracy: {val_acc:.4f}")
        print(f"Batch labels mean: {labels.mean():.2f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_glass_break_model.pth")
            
    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.4f}")
    return model


if __name__ == "__main__":
    DATA_DIR = "./dataset"
    LABEL_MAP = {"normal": 0, "glass_breaking": 1}
    
    if os.path.exists(DATA_DIR):
        train_data, val_data, test_data = prepare_data_pipeline(DATA_DIR, LABEL_MAP)
        
        train_loader = DataLoader(AudioDataset(*train_data), batch_size=16, shuffle=True)
        val_loader = DataLoader(AudioDataset(*val_data), batch_size=16, shuffle=False)
        
        sample_spec = train_data[0][0]
        model = BaselineAudioCNN(input_height=sample_spec.shape[0], input_width=sample_spec.shape[1])
        
        trained_model = train_and_evaluate_model(model, train_loader, val_loader, epochs=10)
    else:
        print(f"Directory '{DATA_DIR}' not found. Please organize your audio subdirectories ('normal' and 'glass_breaking') to run the pipeline.")
