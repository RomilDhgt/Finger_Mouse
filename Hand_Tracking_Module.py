# import necessary libraries
import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, dCon=0.5, tCon=0.5):
        # Initializing variables with the input parameters
        self.mode = mode
        self.maxHands = maxHands
        self.dCon = dCon
        self.tCon = tCon
        
        self.tipId = [4, 8, 12, 16, 20]

        # Initializing necessary mediapipe objects 
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode = self.mode, max_num_hands=  self.maxHands, min_detection_confidence = dCon, min_tracking_confidence = tCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findLength(self, finger1, finger2, img):
        x1, y1, = self.lmList[finger1][1], self.lmList[finger1][2]
        x2, y2, = self.lmList[finger2][1], self.lmList[finger2][2]
        cx ,cy = (x1+x2)//2 , (y1+y2)//2

        cv2.circle(img, (x1,y1), 15, (255,0,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255,0,0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (255,0,0), 3)
        cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)

        if length < 50:
            cv2.circle(img, (cx,cy), 15, (0,0,255), cv2.FILLED)
        
        return length, img

    def findHands(self, img, draw = True):
        
        # Converting to the img that is read from the camera to a RGB image for the hands object to use
        imgRBG = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Getting results by processing the cameras RGB image through the hands object
        self.results = self.hands.process(imgRBG)
        # If a hand is detected
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlm, self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPos(self, img, handNo = 0, draw = True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            # Getting landmark data, id and the position 
            for id, pos in enumerate(myHand.landmark):
                # Getting the height width and the center of the image being read in 
                h, w, c = img.shape
                # Getting the x, y position of our landmarks 
                cx , cy = int(pos.x * w) , int(pos.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255,255,0), cv2.FILLED)
        
        return self.lmList

    def fingersUp(self):
        fingers = []
        
        if len(self.lmList) != 0:
            # This if statement is pertaining to the thumb
            if self.lmList[self.tipId[0]][1] < self.lmList[self.tipId[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # This if statement is pertaining to the four fingers
            for id in range(1,5):
                if self.lmList[self.tipId[id]][2] < self.lmList[self.tipId[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return fingers

def main():

    pastTime = 0
    currentTime = 0
    lmList = []

    # Creating the video object using my laptops video camera
    cap = cv2.VideoCapture(0)

    # Creating hand detector object
    detector = handDetector()

    while True:
        # Getting the video from the camera
        success, img = cap.read()
        # Using the findHands() function in the detector class
        img = detector.findHands(img)

        lmList = detector.findPos(img)

        if len(lmList) != 0 :
            print(lmList[4])
        # Getting timestamps to be able to use the trackers positioning data
        currentTime = time.time()
        framesPerSec = 1/(currentTime-pastTime)
        pastTime = currentTime
        # Displaying the frames per second the the screen
        cv2.putText(img,str(int(framesPerSec)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        # Showing the video from camera in a window
        cv2.imshow("Image",img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()