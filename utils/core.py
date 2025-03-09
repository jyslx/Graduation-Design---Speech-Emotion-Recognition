import os
import time
import wave
import numpy
import pyaudio
import soundfile as sf
from functools import wraps
from pydub import AudioSegment

def convert_mp3_to_wav(mp3_file: str, output_wav_file: str) -> None:
    """
    单文件将MP3 转换为 WAV

    :param mp3_file: MP3文件路径
    :param optput_wav_file: 输出WAV路径
    """
    import pyaudio
    from pydub import AudioSegment
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(output_wav_file, format="wav")
        print(f" 转换成功: {mp3_file} -> {output_wav_file}")
    except Exception as e:
        print(f" 转换失败: {e}")


def batch_convert_mp3_to_wav(input_folder: str, output_folder: str) -> None:
    """
    批量转换MP3为WAV

    :param input_folder: MP3文件夹
    :param output_folder: WAV文件夹
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            mp3_file = os.path.join(input_folder, file)
            wav_file = os.path.join(output_folder, file.replace(".mp3", ".wav"))
            convert_mp3_to_wav(mp3_file, wav_file)


def play_audio(file_path: str) -> None:
    """
    播放语音

    :param file_path: 音频文件路径
    """
    p = pyaudio.PyAudio()
    f = wave.open(file_path, "rb")
    stream = p.open(
        format=p.get_format_from_width(f.getsampwidth()),
        channels=f.getnchannels(),
        rate=f.getframerate(),
        output=True,
    )
    data = f.readframes(f.getnframes()[3])
    stream.write(data)
    stream.stop_stream()
    stream.close()
    f.close()


def check_wav_channels_sf(file_path: str) -> None:
    """
    判断WAV音频的声道

    :param file_path:  音频文件
    """
    data, samplerate = sf.read(file_path)
    num_channels = data.shape[1] if len(data.shape) > 1 else 1  # 判断通道数
    print(f" 该 WAV 文件是 {num_channels} 声道")



def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"开始执行: {func.__name__}")

        result = func(*args, **kwargs)  # 调用原始函数

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} 执行完成，耗时: {elapsed_time:.2f} 秒")

        return result  # 返回原始函数的返回值（如果有）

    return wrapper
if __name__ == '__main__':
    # 示例
    check_wav_channels_sf("your_audio.wav")
