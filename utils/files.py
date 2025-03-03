import os
import sys
import numpy as np
from random import shuffle
from typing import Tuple, Union


def remove(file_path: str) -> None:
    """
    批量删除指定路径下所有非wav文件

    :param file_path(str): 文件路径
    """
    for root, dirs, files in os.walk(file_path):  # 用于递归遍历 file_path 目录下的 所有子目录及文件
        for item in files:
            if not item.endswith('.wav'):
                print("Delete file: ", os.path.join(root, item))
                os.remove(os.path.join(root, item))


def rename(file_path: str) -> None:
    """
    批量按照指定格式重命名不同感情音频文件

    :param file_path(str): 文件路径
    """
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if item.endswith('.wav'):
                people_name = root.split('/')[-2]
                emotion_name = root.split('/')[-1]
                item_name = item[:-4]
                old_path = os.path.join(root, item)
                new_path = os.path.join(root, item_name + '-' + emotion_name + '-' + people_name + '.wav')  # 新音频路径
                try:
                    os.rename(old_path, new_path)
                    print('converting', old_path, ' to ', new_path)
                except:
                    continue


def mkdirs(folder_path: str) -> None:
    """
    检查文件夹是否存在，如果不存在就创建一个

    :param file_path(str): 文件路径
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete(directory: str, extensions: list) -> None:
    """
    删除指定路径下的所有指定扩展名的文件

    :param file_path(str): 文件路径
    :param extensions(list): 需要删除文件扩展名列表
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete: {file_path} : {e}")


def get_data_path(data_path: str, class_labels: list) -> list:
    """
    获取所有音频路径

    :param data_path: 数据集文件夹路径
    :param class_labels: 情感标签
    """
    wav_file_path = []

    cur_dir = os.getcwd()  # 获取当前文件夹路径
    sys.stderr.write("Curdir: %s\n" % cur_dir)
    os.chdir(data_path)

    # 遍历文件夹
    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        # 读取该文件夹下的音频
        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path


def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    提取音频特征

    :param config:
    :param train:
    """
    pass


if __name__ == '__main__':
    pass
