import streamlit as st
import numpy as np
import os
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

model_path = '/home/santenana/Proyectos/02_ObjectDetection/Front/best.pt'

st.markdown("# Detection Image")

try:
    model = YOLO(model_path)
    st.success("Model Loaded")
except Exception as e:
    st.error(f"Error in Model Load: {e}")

# def read_image_file(path):
#     img = Image.open(path)
#     img = img.convert('RGB')
#     img_array = np.array(img)
#     return img_array

def Video(path_video,best_model):
    video = cv2.VideoCapture(path_video)  
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('deteccion.avi',fourcc,30,(w,h))  
    labels = best_model.model.names 
    all_predicted_labels = []
    while video.isOpened():  
        ret,frame = video.read()
        if ret == False:
            break
        cv2.imshow("", frame)
        res = best_model.predict(frame,imgsz=640) 
        predicted_labels = []
        for result in res:
            for pred in result.boxes:
                label_index = int(pred.cls) 
                label = labels[label_index]
                predicted_labels.append(label)
        all_predicted_labels.append(predicted_labels)
        anote = res[0].plot()
        out.write(anote)
        cv2.imshow("", anote)         
        if cv2.waitKey(1) == 27: 
            break
        sleep(1/60)
    video.release()
    label = all_predicted_labels
    f_label = [item for sublist in label for item in sublist]
    label_counts = Counter(f_label)
    total_count = sum(label_counts.values())
    label_probabilities = {key: value / total_count for key, value in label_counts.items()}
    for label, prob in label_probabilities.items():
        label = label
        prob = prob
        # print(f"Clase: {label}, Probabilidad: {prob:.4f}")
    
    cv2.destroyAllWindows()
    return label,prob

def main():
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
        st.session_state.label = None
        st.session_state.proba = None
        st.session_state.pdf_buffer = None  
        st.session_state.file_extension = None
        st.session_state.imagen = None
        st.session_state.predicted_labels = None
        st.session_state.show_initial_image = True
        st.session_state.path_img = None
    st.session_state.path_img = st.file_uploader("📁 Load Image", type=["mp4", "avi", "mov"])
    
    if st.session_state.path_img is not None and st.session_state.show_initial_image:
        file_extension = os.path.splitext(st.session_state.path_img.name)[1].lower()
        st.session_state.image_array = read_image_file(st.session_state.path_img)
        # st.image(st.session_state.image_array, caption="Image Loaded", use_column_width=True)
        # st.session_state.image_array = None

    if st.button("Detect Accident Severity"):
        st.session_state.imagen, st.session_state.predicted_labels  = Video(st.session_state.path_img)
        st.image(st.session_state.imagen, caption="Crash Detection", use_column_width=True)
        valor_predicho = st.session_state.predicted_labels[0]
        st.text_input("Prediction", value=valor_predicho, disabled=True)
        
    if st.button("🔄 Reiniciar Aplicación"):
        st.rerun()
    
    if st.button("🔙"):
        st.switch_page('/home/santenana/Proyectos/02_ObjectDetection/Front/front.py')
if __name__ == "__main__":
    main()