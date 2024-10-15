import streamlit as st
import utilidades as util

st.markdown("# Detection Video")

# st.sidebar.markdown("# Welcome")
# st.sidebar.markdown("# Detection Image 🖼️")
# st.sidebar.markdown("# Detection Video 📽️")
# st.sidebar.markdown("# About Model")
# st.sidebar.markdown("# About SGS Insurance")

# st.markdown("""
#     <style>
#     h1 {
#         font-size: 5vw; /* 5% del ancho de la pantalla */
#         text-align: center; /* Centrado */
#         width: 100%; /* Ocupa todo el ancho */
#         color: #4CAF50; /* Color del título */
#     }

#     /* Ajustes para pantallas más pequeñas */
#     @media (max-width: 1080px) {
#         h1 {
#             font-size: 8vw;}
#         }
#     </style>
#     """, unsafe_allow_html=True)
# st.markdown("<h1>🚗 Car-State Accident Detection 🚨</h1>", unsafe_allow_html=True)


# model_path = "/home/santenana/Proyectos/02_ObjectDetection/best.pt"
# try:
#     model = torch.load(model_path)
#     st.success("🧠 Modelo cargado exitosamente")
# except Exception as e:
#     st.error(f"Error al cargar el modelo: {e}")


# def read_video_file(path,modelo):
#     Video(path,modelo)

# # Función para leer archivos JPG o PNG
# def imagen_detect(path,model):
#     res = model.predict(path,imgsz=640)
#     imagen = res[0].plot()
#     labels = res[0].names
#     predicted_labels = []
#     for result in res:
#         for pred in result.boxes:
#             label_index = int(pred.cls)
#             label = labels[label_index]
#             predicted_labels.append(label)
#     return imagen,predicted_labels

# def main():
#     if 'image_array' not in st.session_state:
#         st.session_state.image_array = None
#         st.session_state.label = None
#         st.session_state.proba = None
#         st.session_state.heatmap = None
#     st.title("")
#     # Cargar array
#     uploaded_file = st.file_uploader("🖼️ Cargar Imágen del accidente", type=["dcm", "jpg", "jpeg", "png"])
#     uploaded_file = st.file_uploader("📽️ Cargar Video del accidente", type=["mp4", "avi"])
    
#     if uploaded_file is not None:
#         file_extension = os.path.splitext(uploaded_file.name)[1].lower()
#         if file_extension == ".dcm":
#             st.session_state.image_array = read_video_file(uploaded_file)
#         else:
#             st.session_state.image_array = imagen_detect(uploaded_file)
        
#         # Mostrar array original
#         st.image(st.session_state.image_array, caption="🖼 Imágen Radiográfica cargada", use_column_width=True)

#         if st.button("🤖 Predecir"):
#             st.session_state.label, st.session_state.proba, st.session_state.heatmap = predict(st.session_state.image_array)
            
#             # Mostrar resultados
#             st.write(f"Zona afectada: {st.session_state.label}")
#             st.write(f"Probabilidad de lesion crítica: {st.session_state.proba:.2f}%")
            
#             # Mostrar heatmap
#             st.image(st.session_state.heatmap, caption="🔥 Imágen Radiográfica con zonas afectadas", use_column_width=True)


#     # Botón para reiniciar la aplicación
#     if st.button("🔄 Reiniciar Aplicación"):
#         st.rerun()

# if __name__ == "__main__":
#     main()