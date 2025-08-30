import cv2
import mediapipe as mp
import time

class FaceMesh():
    def __init__(self,mode = False, num_of_faces = 2, detection_confidence= 0.5,tracking_confidence = 0.5):
        self.mode = mode
        self.num_of_faces = num_of_faces
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.faceMeshDetection = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceDetector = self.faceMeshDetection.FaceMesh(static_image_mode=self.mode,max_num_faces=self.num_of_faces,min_detection_confidence=self.detection_confidence,min_tracking_confidence=self.tracking_confidence)
        self.connections = [self.faceMeshDetection.FACEMESH_CONTOURS, self.faceMeshDetection.FACEMESH_FACE_OVAL,
               self.faceMeshDetection.FACEMESH_LIPS, self.faceMeshDetection.FACEMESH_LEFT_EYE,
               self.faceMeshDetection.FACEMESH_RIGHT_EYE,
               self.faceMeshDetection.FACEMESH_LEFT_EYEBROW,
               self.faceMeshDetection.FACEMESH_RIGHT_EYEBROW,
               self.faceMeshDetection.FACEMESH_NOSE]


    def findFaces(self,img,draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.mpFaceDetector.process(imgRGB)
        landMarkList = []
        #if face is present
        if self.results.multi_face_landmarks:
            # for each facial landmark in the faces, draw
            for each_Face in self.results.multi_face_landmarks:
                # loop through each connections to draw
                for conn in self.connections:
                    if draw:
                        self.mpDraw.draw_landmarks(img, each_Face, conn, self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=4))
                    face = []
                for index,landmark in enumerate(each_Face.landmark):
                    imageHeight, imageWidth,imageChannel = img.shape
                    x, y = int(landmark.x * imageWidth), int(landmark.y * imageHeight)
                    #cv2.putText(img,str(index),(x,y),cv2.FONT_HERSHEY_PLAIN,2,(0,0,0),1)
                    face.append([x,y])
                landMarkList.append(face)
        return img,landMarkList




def main():
    #load the video frame
    cap = cv2.VideoCapture("Videos/Video2.mp4")
    previousTime = 0
    detector = FaceMesh()
    while True:
        #read the video
        success,img = cap.read()

        if not success:
            break
        img,faces = detector.findFaces(img)
        if len(faces)!=0:
            print(len(faces))
        #for frame per second
        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(img,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),3)
        cv2.imshow("Image",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == "__main__":
    main()