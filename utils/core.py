import os
import soundfile as sf

def convert_mp3_to_wav(mp3_file: str, optput_wav_file: str) -> None:
    """
    单文件将MP3 转换为 WAV

    :param mp3_file: MP3文件路径
    :param optput_wav_file: 输出WAV路径
    """
    pass



def batch_convert_mp3_to_wav(input_folder: str, output_folder: str) -> None:
    """
    批量转换MP3为WAV

    :param input_folder: MP3文件夹
    :param output_folder: WAV文件夹
    """
    pass


def play_audio(file_path: str) -> None:
    """
    播放语音

    :param file_path: 音频文件路径
    """
    pass


def check_wav_channels_sf(file_path: str) -> None:
    """
    判断WAV音频的声道

    :param file_path:  音频文件
    """
    data, samplerate = sf.read(file_path)
    num_channels = data.shape[1] if len(data.shape) > 1 else 1  # 判断通道数
    print(f" 该 WAV 文件是 {num_channels} 声道")

if __name__ == '__main__':
    # 示例
    check_wav_channels_sf("your_audio.wav")
