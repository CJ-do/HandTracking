#import all the libraries
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode = False,maxHands = 2, detectionCon= 0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # mpHands now contains all the classes and methods for the Hands solution
        self.mpHands = mp.solutions.hands
        # mpDraw now contains all the classes and methods of drawing and utilities solution
        self.mpDraw = mp.solutions.drawing_utils
        # hands is a variable and it holds the instance of Hands() class
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon)

    def findHands(self,img,draw =True):
        if img is None:
            return None
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # process is a method inside Hands() class that takes the RGB image and classifies.
        self.results = self.hands.process(imgRGB)
        #  multi_hand_landmarks contains a list of 21 landmarks(wrist,Thumb,finger and so on)  for each detected hand'
        if self.results.multi_hand_landmarks:
            # for each detected hand in the frame
            for eachHand in self.results.multi_hand_landmarks:
                if draw:
                    # if multi_hand_landmarks is present, then draw the detected handmark and its connection
                    self.mpDraw.draw_landmarks(img, eachHand, self.mpHands.HAND_CONNECTIONS)
        return img



    def findPosition(self,img, handNo=0, draw = True, colorLandMark = None):
        #create a landMarkPosition list that stores position of all hand landmark detected
        landmarkPosition = []
        #if hand is detected
        if self.results.multi_hand_landmarks:
            #myHand contains all 21 landmark for that handNumber
            myHand = self.results.multi_hand_landmarks[handNo]
            #loops through 21 landmarks of that hand number
            for index, coordinate in enumerate(myHand.landmark):
                #gets the size of the image
                height, width, channel = img.shape
                #converts into pixel
                cx, cy = int(coordinate.x * width), int(coordinate.y * height)
                #adds the landmarks index and pixel coordinate to the list
                landmarkPosition.append([index,cx,cy])
                if draw:
                    if colorLandMark is not None and index == colorLandMark:
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return landmarkPosition









def main():
    previousTime = 0
    currentTime = 0
    # create a video capture object and select the web camera that is being used
    cap = cv2.VideoCapture(1,cv2.CAP_ANY)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        exit()

    detector = handDetector()

    # loop and keep reading from the webcam until its stopped.
    while True:
        # success returns, whether the frame was read and img is actual image from the webcam
        success, img = cap.read()

        if not success or img is None:
            print("❌ Failed to capture frame")
            continue

        image = detector.findHands(img)
        landMarkList = detector.findPosition(image)
        if len(landMarkList) !=0:
            #prints the index and coordinate pixel of the 4th landmark
            print(landMarkList[4])

        if image is None or image.size == 0:
            continue  # skip invalid frame

        #'The below code shows how many frame can the program process and show which means reading the frame processing and displaying '
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


if __name__ == "__main__":
    main()