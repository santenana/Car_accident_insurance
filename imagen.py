import pandas as pd
import numpy as np
import cv2 as cv
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

##best_model = YOLO("/home/santenana/Proyectos/02_ObjectDetection/runs/detect/train12/weights/best.pt")

def imagen_detect(path,model):
    res = model.predict(path,imgsz=640)
    imagen = res[0].plot()
    ##imagen_2 = cv2.imwrite("/home/santenana/Proyectos/02_ObjectDetection/Deteccion4.jpg",imagen)
    labels = res[0].names
    predicted_labels = []
    for result in res:
        for pred in result.boxes:
            label_index = int(pred.cls)
            label = labels[label_index]
            predicted_labels.append(label)
    # predicted_labels
    return imagen,predicted_labels