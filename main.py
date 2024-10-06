import pandas as pd
import numpy as np  
import cv2 as cv
import torch 
from ultralytics import YOLO 
import matplotlib.pyplot as plt
import cv2

##best_model = YOLO("/home/santenana/Proyectos/02_ObjectDetection/runs/detect/train12/weights/best.pt")

def imagen(path,model):
    res = model.predict(path,imgsz=640) 
    print(res.names)
    imagen = res[0].plot()
    imagen_2 = cv2.imwrite("/home/santenana/Proyectos/02_ObjectDetection/Deteccion4.jpg",imagen)
    return imagen_2