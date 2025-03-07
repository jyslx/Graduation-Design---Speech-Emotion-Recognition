import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from base import BaseModel
from abc import ABC, abstractmethod
from models import SERDataset, DataProcessor
from tqdm import tqdm
from extract_feats import Librosa
from extract_feats.Librosa import *
from utils import config, files, plot

feature = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\features\librosa\feature.csv"


class SERLSTM(BaseModel, nn.Module):
    def __init__(self, config):
        # 先初始化 BaseModel 和 nn.Module
        BaseModel.__init__(self, model=None, config=config)  # 需要初始化Base类的构造参数
        nn.Module.__init__(self)
        self.lstm = nn.LSTM(
            input_size=self.config.input_size,  # 输入特征的维度（如MFCC特征数）
            hidden_size=self.config.hidden_size,  # 隐藏层神经元数量（如128）
            num_layers=self.config.num_layers,  # LSTM堆叠层数（如2层）
            bidirectional=True,  # 启用双向LSTM
            batch_first=True  # 输入格式为(batch, seq, feature)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(0.3)  # 随机丢弃30%神经元防止过拟合
        self.fc = nn.Linear(self.config.hidden_size * 2, self.config.num_classes)  # 双向需将隐藏层维度乘2
        self.label_encoder = None
        self.scaler = None

    def forward(self, x):
        # 输入x形状: (batch_size, seq_len=1, input_size)
        out, _ = self.lstm(x)  # 输出形状: (batch_size, seq_len, hidden_size*2)
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步的输出
        out = self.fc(out)  # 全连接层分类
        return out

    def train_model(self):

        if self.trained:
            processor = DataProcessor(self.config)  # 数据预处理管道
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.load_and_preprocess()

            # 保存标签编码器和标准化器
            self.label_encoder = processor.label_encoder
            self.scaler = processor.scaler  # 关键：存储scaler

            train_dataset = SERDataset(X_train, y_train)
            val_dataset = SERDataset(X_val, y_val)
            test_dataset = SERDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        """执行模型训练"""
        best_acc = 0.0
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(self.config.epochs):
            train_loss, train_correct = 0.0, 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * labels.size(0)
                train_correct += (outputs.argmax(1) == labels).sum().item()

            train_loss /= len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)

            val_loss, val_correct = 0.0, 0
            self.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)

                    val_loss += self.criterion(outputs, labels).item() * labels.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / len(val_loader.dataset)

            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f" Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}\n")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                self.save()
        # 训练完成后绘制曲线
        plot.curve(train_losses, val_losses, "Loss Curve", "Loss")
        plot.curve(train_accuracies, val_accuracies, "Accuracy Curve", "Accuracy")

    def save(self):
        files.mkdirs(self.config.checkpoint_path)
        torch.save(self.state_dict(), self.config.checkpoint_path + self.config.checkpoint_name)

    def predict(self, input_data: str):
        """
        预测音频情感类别，并返回置信度分布

        :param input_data: 音频文件路径
        :return: (预测类别, 置信度字典)
        """
        # 1. 特征提取
        features = eval(self.config.feature_method).extract_features(input_data)

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


if __name__ == '__main__':
    testwav = r"C:\Users\35055\Desktop\example.wav"
    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\demo.ini"
    config = config.get_config(ini_path)
    lstm = SERLSTM(config)
    lstm.train_model()
    lstm.predict(testwav)
