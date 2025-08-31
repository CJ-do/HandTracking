#import libraries
import cv2
import time
import numpy as np
import HandTrackingModule as ht
import math
import os

def get_volume():
    cmd = "osascript -e 'output volume of (get volume settings)'"
    return int(os.popen(cmd).read().strip())

def set_volume(volume):
    volume = max(0, min(100, volume))
    os.system(f"osascript -e 'set volume output volume {volume}'")



#paramters
#set camera resolution
cameraWidth, cameraHeight = 1280,720
#open camera
cap = cv2.VideoCapture(0)
#set width and height of camera
cap.set(3,cameraWidth)
cap.set(4,cameraHeight)
previousTime =0

detector= ht.handDetector(detectionCon=0.7)
#minimum and maximum distance
minLength = 30 #distance when finger is closer
maxLength =200 #distance when fingers are far apart

while True:
    #read the frame
    success,img = cap.read()
    img= detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        #draw a line between thumb and index finger for volume control
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1], lmList[8][2]
        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        #find the midpoint of the line
        cx,cy = (x1+x2)//2, (y1+y2)//2

        #circle the midpoint of the line
        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        #find the length of the line
        length = math.hypot(x2-x1,y2-y1)
        #Map length to volume
        volume  = np.interp(length,[minLength,maxLength],[0,100])
        set_volume(int(volume))

        #show current volume
        cv2.putText(img, f'Vol: {int(volume)}%', (50, 100),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
        if length<minLength:
            cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)


    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime
    cv2.putText(img, f'FPS{int(fps)}', (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)


    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


