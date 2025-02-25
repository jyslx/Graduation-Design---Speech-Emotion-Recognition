import os
import re
import sys
import numpy as np
import librosa
from typing import Tuple, Union
import matplotlib.pyplot as plt
import librosa.display
from librosa.core import pitch


def features(X, sample_rate: float) -> np.ndarray:
    """
    提取音频数据特征

    :param X: 时域音频数据
    :param sample_rate: 采样率
    """

    stft = np.abs(librosa.stft(X))

    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []  # 音频
    print(magnitudes[:, 1])
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
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # ottava对比
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # 过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.meas(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # 均方根能量
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, mean_cent, std_cent, max_cent, stdMagnitude, pitch_tuning_offset,
        pitch_mean, pit_max, pitch_std, pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))
    return ext_features


def extract_features(audio_path: str, pad: bool = False) -> np.ndarray:
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
    return features(X, sample_rate)




def get_data(config, data_path: str):
    """
    提取说有音频的特征：遍历音频文件夹，提取每一个音频的特征，把所有的特征都存放在feature文件夹中

    :param config: 配置文件
    :param data_path: 数据集文件夹
    """


if __name__ == '__main__':
    audio_path = r"C:\Users\35055\Desktop\example.wav"
    signal = extract_features(audio_path, True)
