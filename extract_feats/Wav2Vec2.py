from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import numpy as np
from utils.files import *
from extract_feats.base import BaseExtractor


class Wav2Vec2Extractor(BaseExtractor):

    def __init__(self, config):
        BaseExtractor.__init__(self, config=config)
        self.config = config
        self.processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
        self.model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
    def features(self, X, sample_rate: float) -> np.ndarray:
        """
        提取音频数据特征

        :param X: 时域音频数据
        :param sample_rate: 采样率
        """

        # 预处理音频数据
        inputs = self.processor(X, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        # 提取wav特征
        with torch.no_grad():
            features = self.model(**inputs).last_hidden_state

        # 取平均池化，得到固定长度特征向量
        features = torch.mean(features, dim=1)

        # 转为numpy
        features_np = features.numpy()

        print(features_np.shape)
        return features_np

    def generate_csv_header(self) -> list:
        """
        设计特征数据信息

        :return: csv列名
        """
        return ["label"] + [f"feature{i}" for i in range(1, 769)]

if __name__ == '__main__':
    file_path = r"C:\Users\35055\Desktop\example.wav"
    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\configuration\new.ini"
    config = config.get_config(ini_path)
    extractor = Wav2Vec2Extractor(config)
    extractor.get_data()
