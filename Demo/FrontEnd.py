import streamlit as st
import matplotlib.pyplot as plt
import BackEnd

from PIL import Image

def Core_Page():

    # nhập Input
    with st.container():
        # Title
        st.title("Trích xuất thông tin từ biển báo")

        # Nhập dữ liệu
        img = st.file_uploader('Đưa ảnh lên')

        # chọn model
        model = st.selectbox('Chọn model', ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'))

    # xuất phần tính toán
    with st.container():

        col1, col2 = st.columns(2)

        with col1:
            try:
                # xuất lại ảnh
                st.header('Ảnh đầu vào : ')
                st.image(Image.open(img))
            except:
                st.header('Chưa có ảnh')

        with col2:
            try:
                lb, point, img_lb = BackEnd.Main(model, img)

                # vẽ label
                st.header('Kết quả : ')
                fig, ax = plt.subplots()
                ax.imshow(img_lb)
                ax.axis('off')
                st.pyplot(fig)
            except:
                st.header('Đang tính toán')
    
def main():
    Core_Page()