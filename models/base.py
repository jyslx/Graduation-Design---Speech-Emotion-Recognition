from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):  # ABC（Abstract Base Class）是 Python 提供的抽象类基类，允许创建抽象类。
    """所有模型的基类"""

    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config
        self.trained = self.config.trained

    @abstractmethod  # 抽象方法，子类必须实现。
    def train_model(self) -> None:
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, data_path: str) -> None:
        """预测音频的情感"""
        pass

    @abstractmethod
    def save(self) -> None:
        """保存模型"""
        pass

    # @classmethod  # 类方法，直接作用于类，而不是实例
    # @abstractmethod
    # def load(cls, path: str, name: str) -> None:
    #     """加载模型"""
    #     pass
    #
    # @classmethod
    # @abstractmethod
    # def make(cls):
    #     """搭建模型"""
    #     pass
