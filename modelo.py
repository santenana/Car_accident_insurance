def modelo_segmentation(dataset,epochs,imgsz,batch):
    import detectron2
    from detectron2.utils.logger import setup_logger
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import random
    import roboflow 
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.data.catalog import DatasetCatalog
    from ultralytics import YOLO
    # model = YOLO("yolov8n-seg.yaml")
    model = YOLO('yolo11n.pt')
    # model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  
    results = model.train(data=dataset, epochs=epochs, imgsz=imgsz,batch=batch)

    best_model = YOLO("/home/santenana/Proyectos/02_ObjectDetection/runs/detect/train/weights/best.pt")
    return best_model


dataset = '/home/santenana/Proyectos/02_ObjectDetection/Car_Accident/data.yaml'
epochs = 25
imgsz = 640
batch = 32
best_model = modelo_segmentation(dataset,epochs,imgsz,batch)
path = '/home/santenana/Proyectos/02_ObjectDetection/best.pt'

torch.save(path)
