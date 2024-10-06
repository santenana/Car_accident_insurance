import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import os
import tempfile
from io import BytesIO
from video_detect import Video

model_path = "/home/santenana/Proyectos/02_ObjectDetection/best.pt"

# Cargar el modelo
try:
    model = torch.load(model_path)
    st.success("🧠 Modelo cargado exitosamente")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")


def read_video_file(path,modelo):
    Video(path,modelo)

# Función para leer archivos JPG o PNG
def imagen_detect(path,model):
    res = model.predict(path,imgsz=640)
    imagen = res[0].plot()
    labels = res[0].names
    predicted_labels = []
    for result in res:
        for pred in result.boxes:
            label_index = int(pred.cls)
            label = labels[label_index]
            predicted_labels.append(label)
    return imagen,predicted_labels

def main():
    # Inicializar el estado de la sesión
    if 'image_array' not in st.session_state:
        st.session_state.image_array = None
        st.session_state.label = None
        st.session_state.proba = None
        st.session_state.heatmap = None
    
    st.title("🚗 Car-State Accident Detection 🚨")

    # Entrada para la identificación del paciente

    patient_id = st.text_input("ID del paciente:")

    # Cargar array
    uploaded_file = st.file_uploader("📁 Cargar Imágen o Videos", type=["dcm", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".dcm":
            st.session_state.image_array = read_video_file(uploaded_file)
        else:
            st.session_state.image_array = imagen_detect(uploaded_file)
        
        # Mostrar array original
        st.image(st.session_state.image_array, caption="🖼 Imágen Radiográfica cargada", use_column_width=True)

        if st.button("🤖 Predecir"):
            st.session_state.label, st.session_state.proba, st.session_state.heatmap = predict(st.session_state.image_array)
            
            # Mostrar resultados
            st.write(f"Zona afectada: {st.session_state.label}")
            st.write(f"Probabilidad de lesion crítica: {st.session_state.proba:.2f}%")
            
            # Mostrar heatmap
            st.image(st.session_state.heatmap, caption="🔥 Imágen Radiográfica con zonas afectadas", use_column_width=True)

            # Generar el PDF solo si se ha realizado una predicción

        
    # Botón de descarga del PDF (fuera del flujo de predicción)
    if st.session_state.pdf_buffer is not None:
        st.download_button(
            label="📄 Descargar Reporte en PDF",
            data=st.session_state.pdf_buffer,
            file_name=f"Diagnóstico{patient_id}.pdf",
            mime="application/pdf"
        )

    # Botón para reiniciar la aplicación
    if st.button("🔄 Reiniciar Aplicación"):
        st.rerun()

if __name__ == "__main__":
    main()