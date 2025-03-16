from utils import config
from utils.core import timing_decorator
from extract_feats import Librosa, Torchaudio
from models import LSTM, RNN

Extractor = {
    'librosaExtractor': Librosa.librosaExtractor,
    'TorchaudioExtractor': Torchaudio.TorchaudioExtractor,
    # 扩展其他特征提取方式
}

_MODELS = {
    'lstm': LSTM.SERLSTM,
    'RNN': RNN.SERRNN,
    # 扩展其他模型，如 CNN, MLP, SVM
}


@timing_decorator
def train(config) -> None:
    """
    训练模型

    :param config: 配置参数
    :return:  返回训练好的模型
    """
    # 第一步对数据集进行特征提取
    extractor = Extractor[config.feature_method](config)
    extractor.get_data()

    # 第二步对特征文件进行模型训练

    # 模型搭建
    model = _MODELS[config.model].load(config)

    # 模型训练

    print("------------ start training ", config.model, " ------------")
    model.train_model()
    print("----------- end training ", config.model, " ------------")


if __name__ == '__main__':
    # ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\demo.ini"
    ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\RNN.ini"
    config = config.get_config(ini_path)
    train(config)
