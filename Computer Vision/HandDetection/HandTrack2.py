import cv2
import mediapipe as mp
import time
import HandTrackingModule as ht

previousTime = 0
currentTime = 0
# create a video capture object and select the web camera that is being used
cap = cv2.VideoCapture(1, cv2.CAP_ANY)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = ht.handDetector()

# loop and keep reading from the webcam until its stopped.
while True:
    # success returns, whether the frame was read and img is actual image from the webcam
    success, img = cap.read()

    if not success or img is None:
        print("Failed to capture frame")
        continue

    image = detector.findHands(img)
    landMarkList = detector.findPosition(image,colorLandMark=4)
    if len(landMarkList) != 0:
        # prints the index and coordinate pixel of the 4th landmark
        print(landMarkList[4])

    if image is None or image.size == 0:
        continue  # skip invalid frame

    # 'The below code shows how many frame can the program process and show which means reading the frame processing and displaying '
    # get the current time in seconds
    currentTime = time.time()
    # take out the frame per second
    fps = 1 / (currentTime - previousTime)
    # update the previousTime
    previousTime = currentTime
    cv2.putText(image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 255), 3)

    # open the window frame called image and display img(current frame)
    cv2.imshow("Image", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
