import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
import os
import csv
from utils.files import *
from extract_feats.base import BaseExtractor


class TorchaudioExtractor(BaseExtractor):
    def __init__(self, config):
        BaseExtractor.__init__(self, config)
        self.config = config
        self.min_audio_length = 512  # 最小音频长度（0.032秒@16kHz）

        self.spec_transform = T.Spectrogram(n_fft=512, hop_length=256, power=None)
        self.mel_transform = None
        self.mfcc_transform = None

    def _compute_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """计算标准化频谱图"""
        spec = torch.abs(self.spec_transform(waveform))
        return (spec - spec.mean()) / (spec.std() + 1e-6)


    def _compute_mel_spectrogram(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """计算梅尔频谱"""
        if self.mel_transform is None or self.mel_transform.sample_rate != sample_rate:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                n_mels=40,
                hop_length=256
            )
        return self.mel_transform(waveform)


    def _compute_mfcc(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """计算MFCC特征"""
        if self.mfcc_transform is None or self.mfcc_transform.sample_rate != sample_rate:
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=20,
                melkwargs={"n_fft": 512, "n_mels": 40, "hop_length": 256}
            )
        return self.mfcc_transform(waveform)


    def generate_csv_header(self) -> list:
        """生成与特征顺序严格对应的CSV头部"""
        return [
            # 时域特征
            "energy_mean", "energy_std", "amp_mean", "amp_range",
            # 频谱特征
            "spec_mean", "spec_std", "flatness", "spectral_centroid",
            # 梅尔特征
            "logmel_mean", "logmel_std", "mel_drange",
            # MFCC特征
            "mfcc_mean", "delta_mfcc", "mfcc_low", "mfcc_high",
            # 基频特征
            "pitch_mean", "pitch_std",
            # 动态特征
            "env_mean", "energy_peak",
            # 标签列
            "label"
        ][::-1]

    def features(self, X: np.ndarray, sample_rate: float) -> np.ndarray:
        """执行完整特征提取流程"""
        waveform = self._preprocess_audio(X)
        return self._extract_features(waveform, sample_rate)

    def _preprocess_audio(self, X: np.ndarray) -> torch.Tensor:
        """音频预处理管道"""
        waveform = torch.as_tensor(X, dtype=torch.float32)

        # 多声道处理
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)

        # 长度标准化
        if waveform.shape[-1] < self.min_audio_length:
            pad_size = self.min_audio_length - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        elif waveform.shape[-1] > 16000:  # 限制最长1秒音频
            waveform = waveform[:16000]

        return waveform.unsqueeze(0)  # 添加批次维度

    def _extract_features(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """核心特征提取逻辑"""
        features = []

        # 时域特征
        features.extend(self._temporal_features(waveform))
        # print("features", features) 4
        # 频谱特征
        spec = self._compute_spectrogram(waveform)
        features.extend(self._spectral_features(spec))

        # 梅尔特征
        mel_spec = self._compute_mel_spectrogram(waveform, sample_rate)
        features.extend(self._mel_features(mel_spec))

        # MFCC特征
        mfcc = self._compute_mfcc(waveform, sample_rate)
        features.extend(self._mfcc_features(mfcc))

        # 基频特征
        features.extend(self._pitch_features(waveform, sample_rate))

        # 动态特征
        features.extend(self._dynamic_features(waveform))

        return np.array(features)

        # 各特征计算方法的实现（与之前提供的相同）

    def _temporal_features(self, waveform: torch.Tensor) -> list:
        energy = torch.mean(waveform ** 2)
        return [
            float(energy),
            float(torch.std(waveform ** 2)),
            float(torch.mean(torch.abs(waveform))),
            float(torch.quantile(waveform, 0.9) - torch.quantile(waveform, 0.1))
        ]

    def _spectral_features(self, spectrogram: torch.Tensor) -> list:
        EPSILON = 1e-12
        device = spectrogram.device

        # 数值稳定性处理
        spectrogram = torch.clamp(spectrogram, min=EPSILON, max=1e12)

        # 几何平均计算（稳定实现）
        log_values = torch.log(spectrogram)
        log_mean = log_values.mean()
        geometric_mean = torch.exp(log_mean - log_values.max() + log_values.min())  # 数值稳定技巧

        # 算术平均
        arithmetic_mean = spectrogram.mean()

        # 平坦度计算
        flatness = geometric_mean / (arithmetic_mean + EPSILON)
        flatness = torch.clamp(flatness, min=EPSILON, max=1.0)

        # 频谱质心（稳定实现）
        weights = torch.arange(1, spectrogram.shape[-1] + 1, device=device)
        weighted_sum = torch.sum(spectrogram * weights)
        total_sum = torch.sum(spectrogram)
        centroid = weighted_sum / (total_sum + EPSILON)

        return [
            float(arithmetic_mean),
            float(spectrogram.std()),
            float(flatness),
            float(centroid)
        ]

    def _mel_features(self, mel_spec: torch.Tensor) -> list:
        log_mel = torch.log(mel_spec + 1e-6)
        return [
            float(log_mel.mean()),
            float(log_mel.std()),
            float(log_mel.max() - log_mel.min())
        ]

    def _mfcc_features(self, mfcc: torch.Tensor) -> list:
        delta_mfcc = torch.diff(mfcc, n=1, dim=-1)
        return [
            float(mfcc.mean()),
            float(delta_mfcc.mean()),
            float(mfcc[:, :5].mean()),
            float(mfcc[:, -5:].mean())
        ]

    def _pitch_features(self, waveform: torch.Tensor, sample_rate: int) -> list:
        try:
            pitch = torchaudio.functional.detect_pitch_frequency(waveform, sample_rate)
            return [float(pitch.mean()), float(pitch.std())]
        except:
            return [0.0, 0.0]  # 降级处理

    def _dynamic_features(self, waveform: torch.Tensor) -> list:
        env = torchaudio.transforms.ComputeDeltas()(waveform)
        energy_peak = torch.max(waveform ** 2) / (torch.mean(waveform ** 2) + 1e-6)
        return [
            float(env.mean()),
            float(energy_peak)
        ]


if __name__ == '__main__':
    audio_path = r"C:\Users\35055\Desktop\example.wav"
    # signal = extract_features(audio_path, True)
    # print(signal)
    # print(signal.shape)

    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\RNN.ini"
    config = config.get_config(ini_path)
    extractor = TorchaudioExtractor(config)
    extractor.get_data()
