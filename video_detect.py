from time import  sleep
import cv2 as cv2
from ultralytics import YOLO 

best_model = YOLO("/home/santenana/Proyectos/02_ObjectDetection/runs/detect/train12/weights/best.pt")

def Video(path_video,best_model):
    
    video = cv2.VideoCapture(path_video) # Se crea la variable que guarda el video 
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter('deteccion.avi',fourcc,30,(w,h))   
    while video.isOpened(): #Se usa para mantener el video abierto 
        ret,frame = video.read()  # Se guardan variables de ret para que el video se mantenga abirto y frame el cual
                              # va guardando cada imagen para su posterior prediccion en segmentacion
        if ret == False:
            break
        cv2.imshow("", frame)
        res = best_model.predict(frame,imgsz=640) # Se hace el predict para cada frame 
    
        anote = res[0].plot() # Se guarda la anotacion del elemento segmentado 
        out.write(anote)
        cv2.imshow("", anote) # Se muetra como imagen la anotacion dentro del video
        
        if cv2.waitKey(1) == 27: # Se usa la tecla de escape para salir del video en caso de que no se quiera
                             # esperar a que el video termine
            break
        sleep(1/60) # Se usa un sleep para que el video tenga una velocidad normal de frames 60fps
    video.release()
    cv2.destroyAllWindows()