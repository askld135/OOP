import cv2
import numpy as np
 
ARUCO_DICT = {
    "DICT_4X4_50" : cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100" : cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250" : cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000" : cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50" : cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100" : cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250" : cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000" : cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50" : cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100" : cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250" : cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000" : cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50" : cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100" : cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250" : cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000" : cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL" : cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICI_APRILTAG_16h5" : cv2.aruco.DICT_APRILTAG_16h5,
    "DICI_APRILTAG_25h9" : cv2.aruco.DICT_APRILTAG_25h9,
    "DICI_APRILTAG_36h10" : cv2.aruco.DICT_APRILTAG_36h10,
    "DICI_APRILTAG_36h11" : cv2.aruco.DICT_APRILTAG_36h11
    }

aruco_type = "DICT_6X6_250"

# Load the predefined dictionary
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, arucoParams) 
 
# Initialize the detector parameters using default values
parameters =  cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, img = cap.read()
    h, w, c = img.shape

    width = 1000
    height = int(width * (h / w))
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
    
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
    cv2.imshow("camera_test", img)

cap.release()
cv2.destroyAllWindows()