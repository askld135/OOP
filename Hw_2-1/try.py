# 참고: "https://velog.io/@huttzza/%EC%8B%A4%EC%8B%9C%EA%B0%84-%EC%96%BC%EA%B5%B4-%EC%9D%B8%EC%8B%9D-%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%A8-1%EC%B0%A8-%EA%B5%AC%ED%98%84"

import cv2 #OpenCV 영상처리

#분류기
#classifier
faceCascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

#video caputure setting
capture = cv2.VideoCapture(0) # initialize, # 내장카메라는 0, 외장은 1부터이다. 컴퓨터에 내장 카메라가 아예 없는 경우, 연결된 외장 카메라는 0
capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280) #CAP_PROP_FRAME_WIDTH == 3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720) #CAP_PROP_FRAME_HEIGHT == 4

#console message
face_id = input('\n enter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look the camera and wait")

count = 0 # # of caputre face images
#영상 처리 및 출력
while True:                                                               # 조건이 항상 참이므로 무한반복
    ret, frame = capture.read() #카메라 상태 및 프레임
    # cf. frame = cv2.flip(frame, -1) 상하반전
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #흑백으로
    faces = faceCascade.detectMultiScale(
        gray,#검출하고자 하는 원본이미지
        scaleFactor = 1.2, #검색 윈도우 확대 비율, 1보다 커야 한다
        minNeighbors = 6, #얼굴 사이 최소 간격(픽셀)
        minSize=(20,20) #얼굴 최소 크기. 이것보다 작으면 무시
    )
    
    #얼굴에 대해 rectangle 출력
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #inputOutputArray, point1 , 2, colorBGR, thickness)
        count += 1
        cv2.imwrite("dataset/User."+str(face_id)+'.'+str(count)+".jpg",gray[y:y+h, x:x+w])
        
    cv2.imshow('image',frame)

	#종료조건
    if cv2.waitKey(1) > 0 : break #키 입력이 있을 때 반복문 종료
    elif count >= 100 : break #100 face sample                          #elif == else if

print("\n [INFO] Exiting Program and cleanup stuff")
    
capture.release() #메모리 해제
cv2.destroyAllWindows()#모든 윈도우 창 닫기