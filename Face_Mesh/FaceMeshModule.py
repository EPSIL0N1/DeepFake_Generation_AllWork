import warnings 
import cv2
import mediapipe as mp # Total 468 points on face
import time

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        # self.staticMode = staticMode
        # self.maxFaces = maxFaces
        # self.minDetectionCon = minDetectionCon
        # self.minTrackCon = minTrackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        # self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)

    def findFaceMesh(self, img, draw=True):
        
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        
        faces = []
        if self.results.multi_face_landmarks: 
            for faceLms in self.results.multi_face_landmarks: 
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # print(id, x, y)
                    cv2.putText(img, f'({x}, {y})', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1) # Raw Points
                    # cv2.putText(img, f'({id})', (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1) # If want to by ID
                    face.append([x, y,])
                faces.append(face)

        return img, faces
    
    
def main():
    cap = cv2.VideoCapture("videos/oneFace.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img, False)
        
        if faces:
            print(faces[0])
            
        cTime = time.time()
        
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        
        cv2.imshow("Image", img)
        frame_count += 1
        
        cv2.waitKey(1)
    print("Frames = ", frame_count)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()