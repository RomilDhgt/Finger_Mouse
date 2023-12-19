import cv2 
import time
import numpy as np
import Hand_Tracking_Module as htm
import mouse
import screeninfo as s

# initializing variables for FPS tracking
pastTime = 0
currentTime = 0

# Finding screen size 
m = str(s.get_monitors())
x , y , width, height, widthmm, heightmm, name, pri = m.split(",")
_, screenWidth = width.split("=")
_, screenHeight = height.split("=")
screenWidth = int(screenWidth)
screenHeight = int(screenHeight)

# Frame size
frameReduction = 200

# Smoothening 
smooth = 5
px, py = 0, 0
cx, cy = 0, 0

# Creating the video capture object using my laptops video camera
cap = cv2.VideoCapture(0)

# Setting the width and height of the camera display
camWidth = 1280
camHeight = 720
cap.set(3,camWidth)
cap.set(4,camHeight)

detector = htm.handDetector(dCon=0.7, tCon=0.7)

while True:
    # Getting the video from the camera
    success, img = cap.read()

    # flip the image
    img = cv2.flip(img,1)

    # Using findHands function in the Hand_Tracking_Module to overlay hand mesh on hand in the image captured by the camera
    img = detector.findHands(img)
    lmList = detector.findPos(img,draw=False)
    
    if len(lmList) != 0:
        # Getting position of the index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x4, y4 = lmList[20][1:]
    
    # using the fingers up method to determine which fingers are up
    fingers = detector.fingersUp()
    
    if len(fingers) != 0:

        cv2.putText(img,f'Mouse Pad:', (40,180), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        cv2.rectangle(img, (frameReduction, frameReduction), (camWidth - frameReduction, camHeight - frameReduction), (255,0,0), 2)

        if fingers[1] == 1 and fingers[2] == 0:

            x3 = np.interp(x1, (frameReduction,camWidth-frameReduction), (0,screenWidth))
            y3 = np.interp(y1, (frameReduction,camHeight-frameReduction), (0,screenHeight))

            cx = px + (x3 - px) / smooth
            cy = py + (y3 - py) / smooth

            mouse.move(cx,cy)
            cv2.circle(img, (x1, y1), 15, (255,0,0), cv2.FILLED)
            px, py = cx, cy
        
        if fingers[1] == 1 and fingers[2] == 1:
            length, img = detector.findLength(8, 12, img)
            if length < 50:
                mouse.click()
        if fingers[1] == 0 and fingers[2] == 0 and fingers[4] == 1:
            cv2.circle(img, (x4, y4), 15, (255,0,0), cv2.FILLED)
            if y4 < camHeight/2:
                mouse.wheel(1)
            elif y4 > camHeight/2:
                mouse.wheel(-1)

    # Getting timestamps to be able to use the trackers positioning data
    currentTime = time.time()
    framesPerSec = 1/(currentTime-pastTime)
    pastTime = currentTime
    # Displaying the frames per second the the screen
    cv2.putText(img,f'FPS:{int(framesPerSec)}', (40,680), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)