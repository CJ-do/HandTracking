#import libraries
import cv2
import time
import mediapipe as mp

class FaceDetectionModule:
    def __init__(self,minDetectionConfidence =0.5):
        self.minimumDetectionConfidence = minDetectionConfidence
        self.mpFace = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetection = self.mpFace.FaceDetection()

    def findFaces(self,img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.mpFaceDetection.process(imgRGB)
        bboxList = []
        if self.results.detections:
            for index,coordinateData in enumerate(self.results.detections):
                boundingBoxData = coordinateData.location_data.relative_bounding_box
                height,width,channel = img.shape
                boundingBoxData_pixels = int(boundingBoxData.xmin*width), int(boundingBoxData.ymin*height),\
                                         int(boundingBoxData.width*width), int(boundingBoxData.height*height)
                bboxList.append([boundingBoxData_pixels,coordinateData.score])
                if draw:
                    img  = self.drawBBox(img,boundingBoxData_pixels)
                    cv2.putText(img, f'{int(coordinateData.score[0] * 100)}%',
                            (boundingBoxData_pixels[0], boundingBoxData_pixels[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 2),
        return img,bboxList

    def drawBBox(self,img,boundingBox, l =50, t =5,rt=1):
        x,y,w,h = boundingBox
        x1,y1 = x+w, y+h
        #draw the bounding box rectangle
        cv2.rectangle(img, boundingBox, (0, 0, 255), rt)
        #give some fancy styles
        #draw a thick line in top left of the bounding box x,y
        cv2.line(img,(x,y),(x+l, y),(0,0,255),t)
        cv2.line(img, (x, y), (x , y+l), (0, 0, 255), t)
        #draw thick line in top right x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 0, 255), t)
        #draw thick line in bottom left x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 0, 255), t)
        # draw thick line in bottom right x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 0, 255), t)

        return img
def main():
    cap = cv2.VideoCapture("Videos/Video1.mp4")
    previousTime = 0
    detector = FaceDetectionModule()
    while True:
        success, img = cap.read()
        img,bbox = detector.findFaces(img)
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 5)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()


