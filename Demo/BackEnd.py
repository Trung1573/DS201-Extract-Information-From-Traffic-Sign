import numpy as np

from ultralytics import YOLO
from PIL import Image

def LoadModel(name):
    # tải model #
    try:
        model = YOLO('./Model/' + name + '_best.pt')
        return model
    except:
        return None

def PredictIMG(name, img):
    # Trả về kết quả #

    # model
    model = LoadModel(name)
    
    # Predict
    try:
        return model.predict(Image.open(img), save = False)
    except:
        return None
    
def ProcessResult(result):
    # đọc kết quả #

    lb = result.boxes.cls
    point = result.boxes.xywh

    img_lb = np.array(result.plot()) / 255. 

    return lb, point, img_lb

def Main(name, img):
    result = PredictIMG(name, img)

    try:
        return ProcessResult(result[0])
    except:
        return None