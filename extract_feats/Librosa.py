import os
import re
import sys
from utils.files import *
import csv
import numpy as np
import librosa
from typing import Tuple, Union
import matplotlib.pyplot as plt
import librosa.display
from librosa.core import pitch
import ast
from extract_feats.base import BaseExtractor


class librosaExtractor(BaseExtractor):

    def __init__(self, config):
        BaseExtractor.__init__(self, config=config)
        self.config = config


    def features(self, X, sample_rate: float) -> np.ndarray:
        """
        提取音频数据特征

        :param X: 时域音频数据
        :param sample_rate: 采样率
        """

        stft = np.abs(librosa.stft(X))

        pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
        pitch = []  # 音频
        # print(magnitudes[:, 1])
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()  # 找到当前帧中能量最大的音高
            pitch.append(pitches[index, i])  # 将这一个帧中能量最大的音高作为这一帧的基频

        pitch_tuning_offset = librosa.pitch_tuning(pitches)  # 音调偏移
        pitch_mean = np.mean(pitch)  # 平均基频                                 与性别、说话人特征相关
        pitch_std = np.std(pitch)  # 基频标准差（变化幅度）                        与情绪有关，标准差大，情绪变化大，标准差小，情绪变化小
        pit_max = np.max(pitch)  # 最大基频
        pit_min = np.min(pitch)  # 最小基频

        # 频谱质心  代表的是音频频谱的“重心”，可以理解为频谱的能量中心位置。
        # 它衡量的是高频与低频的相对分布，用于判断声音是明亮（高频多）还是低沉（低频多）
        cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)  # 计算频谱质心
        cent = cent / np.sum(cent)  # 归一化
        mean_cent = np.mean(cent)  # 平均频谱质心
        std_cent = np.std(cent)  # 频谱质心的标准差（变化程度）
        max_cent = np.max(cent)  # 最大频谱质心

        # 谱平面 是衡量频谱的平坦程度的一个特征。它可以用来区分音调清晰的声音（谐波丰富）和噪声成分较多的声音（频谱均匀）。
        flatness = np.mean(librosa.feature.spectral_flatness(y=X))

        # 使用系数为50的MFCC特征
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
        # mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
        # mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
        # 色谱图
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

        # 梅尔频率
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

        # ottava对比
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

        # 过零率
        zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

        S, phase = librosa.magphase(stft)
        meanMagnitude = np.mean(S)
        stdMagnitude = np.std(S)
        maxMagnitude = np.max(S)

        # 均方根能量
        rmse = librosa.feature.rms(S=S)[0]
        meanrms = np.mean(rmse)
        stdrms = np.std(rmse)
        maxrms = np.max(rmse)

        ext_features = np.array([
            flatness, zerocr, meanMagnitude, maxMagnitude, mean_cent, std_cent, max_cent, stdMagnitude,
            pitch_tuning_offset,
            pitch_mean, pit_max, pitch_std, meanrms, maxrms, stdrms
        ])
        # mfccsstd, mfccmax,
        ext_features = np.concatenate((ext_features, mfccs, chroma, mel, contrast))
        print(ext_features.shape)

        # 验证特征维度是否与表头一致
        expected_length = 212
        if len(ext_features) != expected_length:
            raise ValueError(f"特征维度错误: 应为 {expected_length}, 实际为 {len(ext_features)}")

        return ext_features


    def generate_csv_header(self) -> list:
        """
        设计特征数据信息

        :return: csv列名
        """
        header = ["label"]

        # 基础统计特征（16列）
        base_features = [
            "flatness",
            "zerocr",
            "meanMagnitude",
            "maxMagnitude",
            "mean_cent",
            "std_cent",
            "max_cent",
            "stdMagnitude",
            "pitch_tuning_offset",
            "pitch_mean",
            "pit_max",
            "pitch_std",
            "meanrms",
            "maxrms",
            "stdrms",
        ]
        header.extend(base_features)

        # MFCC 特征（50×3=150列）, "std", "max"
        mfcc_labels = ["mfcc_{}_{}".format(i, stat) for stat in ["mean"] for i in range(50)]
        header.extend(mfcc_labels)

        # 色谱图（12列）
        chroma_labels = ["chroma_{}".format(i) for i in range(12)]
        header.extend(chroma_labels)

        # 梅尔频谱（128列）
        mel_labels = ["mel_{}".format(i) for i in range(128)]
        header.extend(mel_labels)

        # 频谱对比度（7列）
        contrast_labels = ["contrast_{}".format(i) for i in range(7)]
        header.extend(contrast_labels)

        return header


if __name__ == '__main__':
    # audio_path = r"C:\Users\35055\Desktop\example.wav"
    # signal = extract_features(audio_path, True)
    # print(signal)
    # print(signal.shape)
    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\demo.ini"
    config = config.get_config(ini_path)
    extractor = librosaExtractor(config)
    extractor.get_data()
