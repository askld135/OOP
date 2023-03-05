import cv2
import mediapipe as mp

#얼굴을 찾고, 찾은 얼굴에 표시를 해주기위한 변수 정의
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    while cap.isOpened():
        succes, image = cap.read()
        if not succes:
            break
        
        
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = face_detection.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detection:
                mp_drawing.draw_detection(image, detection)
                
        
        cv2.imshow("Mediapipe Face Detection", cv2.resize(image, None, fx=0.5, fy=0.5))
        
        if cv2.waitkey(1) == ('q'):
            break
    
cap.release()
cv2.destroyAllWindows()