import cv2
import mediapipe as mp
import time
import FaceDetectionModule as fd

cap = cv2.VideoCapture("Videos/Video2.mp4")
previousTime = 0
detector = fd.FaceDetectionModule()
while True:
    success, img = cap.read()
    img, bbox = detector.findFaces(img)
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break