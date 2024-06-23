from ultralytics import YOLO
from glob import glob
import numpy as np
import cv2
import pyautogui
import matplotlib as plt
class_list = [
    "iris",
    "eye"
]


model = YOLO('best_1900_half_epo.pt')
center_of_iris = None
center_of_eye = None
width, height= pyautogui.size()
aspect_ratio = width / height
focal_length = 30

def perspective_projection(point3d, focal_length, width, height):

    scale = focal_length/ (focal_length + point3d[0][2])   
    pointProj = point3d * scale
    pointNDC = np.array([pointProj[0][0] / aspect_ratio, pointProj[0][1]])
    xScale = 2.0 / width
    yScale = 2.0 / height
    x = ((pointNDC[0] + 1) / xScale - 0.5)
    y = (1 - (pointNDC[1]) / yScale - 0.5) 
    return (x, y)
# Set initial mouse position to the center of the screen

initial_x, initial_y = width // 2, height // 2
pyautogui.moveTo(initial_x, initial_y)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: # 웹캠이 안됐을 때
        break
    detection = model(frame, stream=True)
    eyes = []
    irises = []
    closest_pair = []
    closest_eye = None
    direction_vector = None # 1 frame 1 vector 
    for datas in detection:
        boxes = datas.boxes
        for box in boxes:
            for data in box.data:# xmin, ymin, xmax, ymax, confidence, class
                data = data.tolist()
                conf = data[4]
                num_class = int(data[5])
                if(conf < 0.5):
                    continue                                      
                        
                if (num_class == 1): # eye
                    eyes.append(data)

                else: # xmin, ymin, xmax, ymax, class # iris
                    irises.append(data)

                if len(eyes) == 0 and len(irises) == 0:
                    continue
                
        for iris in irises:
            xmin_iris = iris[0]
            min_distance = float('inf')
            for eye in eyes:
                xmin_eye = eye[0]
                distance = abs(xmin_iris - xmin_eye)
                max_distance = abs(eye[2] - eye[0])
                if min_distance > distance and distance < max_distance:
                    closest_eye = eye

            if closest_eye:
                closest_pair.append((closest_eye, iris))

            closest_eye = None

    if len(closest_pair) > 0:
        eye, iris = closest_pair[0]
        x1, y1, x2, y2 = map(int, eye[:4])
        center_of_eye = (int((x1+x2)/2), int((y1+y2)/2))
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = map(int, iris[:4])
        center_of_iris = (int((x1+x2)/2), int((y1+y2)/2))
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        #cv2.circle(frame, center_of_eye, 5, (0, 0, 255), -1)
        #cv2.circle(frame, center_of_iris, 5, (255, 0, 0), -1)

        vector2d = np.array(center_of_eye) - np.array(center_of_iris)
        vector2d = vector2d * 0.5
        vector3d = np.expand_dims(vector2d, axis=0)
        vector3d = np.append(vector3d, [[1]], axis=1)

        rasterX, rasterY = perspective_projection(vector3d, focal_length, width, height)
        rasterX = max(10, min(rasterX, width-10))
        rasterY = max(10, min(rasterY, height-10))

        pyautogui.moveTo(rasterX, rasterY, duration=0.1)
   
    
    eyes = []
    iris = []
    closest_pair = []
    closest_eye = None


    cv2.imshow('frame', frame)
                
    if cv2.waitKey(1)== ord('q'):
        break

