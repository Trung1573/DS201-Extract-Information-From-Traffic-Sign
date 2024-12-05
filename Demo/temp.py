import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

model = YOLO('./Model/yolov8n_best.pt')

img =  st.file_uploader('Đưa file lên')

try:
    st.image(Image.open(img))
    print(img)
    result = model.predict(Image.open(img), save = False)
    fig, ax = plt.subplots()
    ax.imshow(np.array(result[0].plot()))
    st.pyplot(fig)
except:
    st.header('Chưa có ảnh')