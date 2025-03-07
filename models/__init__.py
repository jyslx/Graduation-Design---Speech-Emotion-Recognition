from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim


class DataProcessor:
    """数据预处理管道"""

    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_and_preprocess(self):
        """加载并预处理数据"""
        # 加载原始数据
        df = pd.read_csv(self.config.feature_folder + self.config.feature_name)
        labels = df["label"].values
        features = df.drop("label", axis=1).values

        # 标签编码
        self.label_encoder.classes_ = np.array(self.config.class_labels)
        encoded_labels = self.label_encoder.transform(labels)

        # 特征标准化
        scaled_features = self.scaler.fit_transform(features)

        # 数据集划分
        X_train, X_temp, y_train, y_temp = train_test_split(
            scaled_features, encoded_labels,
            test_size=0.4,
            stratify=encoded_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.5,
            stratify=y_temp
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class SERDataset(Dataset):
    """语音情感识别数据集"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx].unsqueeze(0), self.labels[idx]


class Validator:

    def __init__(self, config, model, val_loader):
        self.config = config
        self.model = model
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()

    def validate(self):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                total_loss += self.criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        return {
            'loss': total_loss / len(self.val_loader.dataset),
            'accuracy': correct / len(self.val_loader.dataset)
        }

