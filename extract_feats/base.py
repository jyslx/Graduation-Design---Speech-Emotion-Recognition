from abc import ABC, abstractmethod
import numpy as np
import librosa
import librosa.display
import os
from utils.files import *
import csv


class BaseExtractor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def features(self, X, sample_rate: float) -> np.ndarray:
        pass

    @abstractmethod
    def generate_csv_header(self) -> list:
        pass


    def extract_features(self, audio_path: str, pad: bool = False) -> np.ndarray:
        """
        对单个音频文件进行特征的提取

        :param audio_path: 音频文件
        :param pad: 是否需要对音频进行填充
        """
        X, sample_rate = librosa.load(audio_path, sr=None)  # 读取音频文件
        max_ = X.shape[0] / sample_rate  # 计算音频时长

        if pad:
            length = (int(np.ceil(max_) * sample_rate)) - X.shape[0]  # 计算需要填充的时长,向上取整
            X = np.pad(X, (0, int(length)), 'constant')  # 用0进行填充
        return self.features(X, sample_rate)

    def get_data(self) -> None:
        """
        提取说有音频的特征：遍历音频文件夹，提取每一个音频的特征，
        把所有的特征都存放在feature文件夹中

        """
        mkdirs(self.config.feature_folder)
        csv_path = os.path.join(self.config.feature_folder, self.config.feature_name)

        # 1. 获取所有音频文件路径（需要明确 get_data_path 返回的是文件列表还是目录）
        # 假设 get_data_path 返回的是文件路径列表
        audio_files = get_data_path(self.config.data_path, self.config.class_labels)

        # 2. 生成表头
        header = self.generate_csv_header()

        # 3. 一次性打开文件，持续写入
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)  # 写入表头

            for audio_path in audio_files:
                # 4. 提取标签（使用 os.path 确保跨平台兼容性）
                label = os.path.basename(os.path.dirname(audio_path))  # 假设标签是父目录名

                # 5. 提取特征
                try:
                    feature = self.extract_features(audio_path, pad=True)
                    feature_list = feature.flatten().tolist()
                except Exception as e:
                    print(f"处理文件 {audio_path} 失败: {str(e)}")
                    continue

                # 6. 写入数据行
                writer.writerow([label] + feature_list)
