import numpy as np

def curve(train: list, val: list, title: str, y_label: str) -> None:
    """
    绘制损失值和准确率曲线

    :param train: 训练集损失值或准确率数组
    :param val: 测试机损失值或准确率数组
    :param title: 图像标题
    :param y_label:  y轴标题
    """
    pass

def radar(data_prob: np.ndarray, class_labels: list) -> None:
    """
    绘制置信概率雷达图

    :param data_prob: 概率数组
    :param class_labels: 感情标签
    """
    pass

def waveform(file_path: str) -> None:
    """
    绘制音频波形图

    :param file_path: 音频路径
    """
    pass

def spectrogram(file_path: str) -> None:
    """
    绘制频谱图

    :param file_path: 音频路径
    """
    pass