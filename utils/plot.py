import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import librosa
matplotlib.use('TkAgg')  # 使用 Tk 窗口显示

def curve(train: list, val: list, title: str, y_label: str) -> None:
    """
    绘制损失值和准确率曲线

    :param train: 训练集损失值或准确率数组
    :param val: 测试集损失值或准确率数组
    :param title: 图像标题
    :param y_label:  y轴标题
    """

    epochs = range(1, len(train) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train, 'bo-', label="Train")  # 训练数据蓝色
    plt.plot(epochs, val, 'ro-', label="Validation")  # 验证数据红色
    plt.xlabel("Epochs")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


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
    X, sample_rate = librosa.load(file_path, sr=None)
    duration = len(X) / sample_rate  # 计算音频时长
    time_axis = np.linspace(0, duration, len(X))  # 生成时间轴

    plt.figure(figsize=(10, 10))  # 框的大小
    plt.plot(time_axis[:1000], X[:1000])  # 前 1000 个点
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform with Time Axis")
    plt.grid()  # 网格
    plt.show()


def spectrogram(file_path: str) -> None:
    """
    使用短时傅里叶变换（STFT）绘制频谱图

    :param file_path: 音频路径
    """

    # 读取音频
    X, sample_rate = librosa.load(file_path, sr=None)  # 读取音频，保持原采样率

    # 计算 STFT
    # stft_result = librosa.stft(X, n_fft=1024, hop_length=256)  # n_fft 窗口大小（影响频率分辨率）、hop_length 窗口移动步长（影响时间分辨率）
    stft_result = librosa.stft(X)  # n_fft 窗口大小（影响频率分辨率）、hop_length 窗口移动步长（影响时间分辨率）
    # spectrogram = np.abs(stft_result)  # 取绝对值，得到振幅谱
    spectrogram = librosa.amplitude_to_db(abs(stft_result))
    # 绘制频谱图
    plt.figure(figsize=(10, 10))
    # librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, hop_length=256, y_axis="log", x_axis="time")
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(label="Decibels (dB)")
    plt.title("Spectrogram (STFT)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.show()


def pitchForm(file_path: str) -> None:
    """
    基频图

    :param file_path: 音频文件
    """
    # 读取语音文件
    X, sample_rate = librosa.load(file_path, sr=None)  # 读取音频，保留原始采样率
    # 提取基频（F0）
    f0, voiced_flag, voiced_probs = librosa.pyin(X, fmin=50, fmax=500, sr=sample_rate, fill_na=None)

    # 生成时间轴
    times = librosa.times_like(f0, sr=sample_rate)

    # 处理 NaN（线性插值填充）
    f0_interp = np.copy(f0)
    nan_indices = np.isnan(f0)
    if np.any(nan_indices):  # 如果有 NaN 值
        f0_interp[nan_indices] = np.interp(np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices), f0[~nan_indices])

    # 绘制基频曲线
    plt.figure(figsize=(10, 10))
    plt.plot(times, f0, label="Fundamental Frequency (F0)", color="b", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour (Fundamental Frequency)")
    plt.ylim(50, 500)  # 限制基频范围
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    audio_path = r"C:\Users\35055\Desktop\example.wav"
    spectrogram(audio_path)
    # pitchForm(audio_path)
    # waveform(audio_path)
