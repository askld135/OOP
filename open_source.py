import numpy as np
import cv2 as cv
from cv2 import aruco
import datetime
 
cap = cv2.VideoCapture(0)
 
while(True):
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
    parameters = cv.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    print(corners, ids)
    detect = aruco.drawDetectedMarkers(frame, corners, ids)
 
    timestamp = datetime.datetime.now()
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv.putText(detect, ts, (10, 20),
    # cv2.putText(detect, ts, (10, detect.shape[0] - 10),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv.imshow('Detected Aruco Markers', detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()      #실행시킨 창을 닫음
