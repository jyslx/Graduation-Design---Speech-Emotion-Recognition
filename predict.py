from fastapi import FastAPI, UploadFile, File
import shutil
from utils import config
from models import LSTM, RNN
import os
import uvicorn
app = FastAPI()

_MODELS = {
    'lstm': LSTM.SERLSTM,
    'RNN': RNN.SERRNN,
    # 扩展其他模型，如 CNN, MLP, SVM
}

# 读取配置
ini_path = r"C:\Users\35055\Desktop\Graduation-Design---Speech-Emotion-Recognition\configuration\demo.ini"
config = config.get_config(ini_path)

# 加载模型（避免每次请求都重新加载）
model = _MODELS[config.model].load(config)


@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    """ 接收音频文件并进行情感识别预测 """

    # 保存文件到临时目录
    temp_audio_path = f"temp_{file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 进行预测
    prediction = model.predict(temp_audio_path)

    # 删除临时文件
    os.remove(temp_audio_path)

    return {"filename": file.filename, "prediction": prediction}

