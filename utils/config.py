# 主要是读取ini文件

import configparser


class Config:
    """
    dict -> class
    主要是将配置文件中的参数赋值给Config类
    """

    def __init__(self, entries: dict):
        for k, v in entries.items():
            if isinstance(v, dict):
                # 如果值是字典，则递归调用构造函数来展开字典
                self.__dict__.update(Config(v).__dict__)
            else:
                # 否则直接赋值为属性
                self.__dict__[k] = v


def load_parse(config: configparser) -> dict:
    """
    从ConfigParser对象中加载配置参数全部转换为字典

    :config: configparser对象类
    :return: 参数字典
    """
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            config_dict[section][key] = value
    return config_dict


def get_config(file_path: str) -> Config:
    """
    获取配置文件ini，将参数封装为对象

    :param file_path:
    :return: 返回Config对象
    """
    cfg = configparser.ConfigParser()
    cfg.read(file_path, encoding="utf-8")
    config_dict = load_parse(cfg)
    config = Config(config_dict)

    return config


if __name__ == '__main__':
    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\demo.ini"
    config = get_config(ini_path)
    print(config.feature_name)
