import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from models.base import BaseModel
from abc import ABC, abstractmethod
from models import SERDataset, DataProcessor
from tqdm import tqdm
from extract_feats import Librosa
from extract_feats.Librosa import *
from extract_feats.Wav2Vec2 import *
from extract_feats.Torchaudio import *
from utils import config, files, plot
import joblib
import wandb


class SERCNN(BaseModel, nn.Module):
    def __init__(self, config):
        # 先初始化 BaseModel 和 nn.Module
        BaseModel.__init__(self, model=None, config=config)  # 需要初始化Base类的构造参数
        nn.Module.__init__(self)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 降维

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 降维

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 降维
        )

        self.dropout = nn.Dropout(0.3)  # 随机丢弃30%神经元防止过拟合
        self.fc = nn.Linear(64, self.config.num_classes)

        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.to(self.device)  # 迁移到 GPU

        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr)

        # 全局平均池化 + 全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        for param in self.optimizer.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(self.device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(self.device)

        self.label_encoder = None
        self.scaler = None

        wandb.init(
            project="SER",
            name="CNN",
            config={
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                # "device": self.device,
                "dropout": self.config.dropout,
                "model": "CNN"
            }
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.to(self.device)
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)  # 全局平均池化
        x = x.view(x.size(0), -1)  # 展平成 (batch, channels)
        x = self.fc(x)  # 全连接层
        return x

    def train_model(self):
        if self.trained:
            processor = DataProcessor(self.config)  # 数据预处理管道
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.load_and_preprocess()
            self.label_encoder = processor.label_encoder
            self.scaler = processor.scaler  # 关键：存储scaler

            train_dataset = SERDataset(X_train, y_train, device=self.device)
            val_dataset = SERDataset(X_val, y_val, device=self.device)
            test_dataset = SERDataset(X_test, y_test, device=self.device)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        """执行模型训练"""
        best_acc = 0.0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(self.config.epochs):
            self.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                self.optimizer.zero_grad()
                print("输入input的形状 ", inputs.shape) # 输入input的形状  torch.Size([10, 1, 1, 19])
                outputs = self(inputs)
                print("debug1")
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * labels.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            self.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    val_loss += self.criterion(outputs, labels).item() * labels.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f" Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            wandb.log({
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_acc": float(train_acc),
                "val_acc": float(val_acc),
            })

            if val_acc > best_acc:
                best_acc = val_acc
                self.save()

        # 训练完成后绘制曲线
        plot.curve(train_losses, val_losses, "Loss Curve", "Loss")
        plot.curve(train_accuracies, val_accuracies, "Accuracy Curve", "Accuracy")

    def save(self):
        files.mkdirs(self.config.checkpoint_path)
        torch.save(self.state_dict(), self.config.checkpoint_path + self.config.checkpoint_name)

    def predict(self, data_path: str) -> str:
        """
        预测音频情感

        :param data_path: 音频文件路径
        :return: 返回情感类别
        """
        self.to(self.device)
        self.eval()  # 设置为评估模式

        # 1. 加载训练好的权重
        checkpoint_path = self.config.checkpoint_path + self.config.checkpoint_name
        self.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        # 2. 还原 LabelEncoder 和 Scaler
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(self.config.class_labels)
        self.scaler = joblib.load(self.config.feature_folder + "scaler.pkl")  # 加载标准化模型

        # 3. 特征提取
        features = eval(self.config.feature_method).extract_features(data_path)

        # 4. 特征标准化（使用训练时的scaler）
        if not hasattr(self, 'scaler'):
            raise ValueError("Scaler not found! Model must be trained first.")
        scaled_features = self.scaler.transform(features.reshape(1, -1))  # 保持二维形状

        # 5. 转换为 Tensor 并匹配模型输入维度
        input_tensor = torch.FloatTensor(scaled_features).unsqueeze(1).to(self.device)  # 形状: (1, 1, 313)

        # 不手动调用 forward，直接调用模型
        with torch.no_grad():
            outputs = self(input_tensor)  # 这里自动调用 forward()
            probabilities = torch.softmax(outputs, dim=1)  # 计算类别概率

        # 7. 解析预测结果
        confidence = {
            self.label_encoder.classes_[i]: float(prob)
            for i, prob in enumerate(probabilities.squeeze().cpu().numpy())
        }
        predicted_class = max(confidence, key=confidence.get)  # 获取最高置信度的类别

        return predicted_class

    def predict_proba(self, input_data: str):
        """
        预测音频情感类别，并返回置信度分布

        :param input_data: 音频文件路径
        :return: (预测类别, 置信度字典)
        """
        # 1. 特征提取
        extractor = eval(self.config.feature_method)(self.config)
        features = extractor.extract_features(input_data)

        # 2. 特征标准化（使用训练时的scaler）
        if not hasattr(self, 'scaler'):
            raise ValueError("Scaler not found! Model must be trained first.")
        scaled_features = self.scaler.transform(features.reshape(1, -1))  # 注意保持二维形状

        # 3. 转换为Tensor并匹配输入维度
        input_tensor = torch.FloatTensor(scaled_features) \
            .unsqueeze(1) \
            .to(self.device)  # 形状: (1, 1, 313)

        # 4. 模型推理
        self.eval()
        with torch.no_grad():
            outputs = self(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # 5. 转换为字典形式的置信度
        confidence = {
            self.label_encoder.classes_[i]: float(prob)
            for i, prob in enumerate(probabilities.squeeze().cpu().numpy())
        }
        print(confidence)

        predicted_class = max(confidence, key=confidence.get)  # 最大概率情绪

        data_prob = np.array(list(confidence.values()))  # 转换为 NumPy 数组
        class_labels = list(confidence.keys())  # 获取类别标签
        plot.radar(data_prob, class_labels)  # 绘制雷达图

    @classmethod
    def load(cls, config):
        model = SERCNN(config)
        return model


if __name__ == '__main__':
    testwav = r"C:\Users\35055\Desktop\example.wav"
    # ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\configuration\demo.ini"
    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\configuration\RNN.ini"
    # ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\configuration\new.ini"
    config = config.get_config(ini_path)
    CNN = SERCNN(config)
    CNN.train_model()
    CNN.predict_proba(testwav)
