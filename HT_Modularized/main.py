# To set up the environment, in your terminal (Windows), do: 
# pip install opencv-python
# pip install mediapipe

import cv2
import mediapipe as mp
import time
import HT_functionalities as HD

# Frame rate calculation variables
prevTime = 0
curTime = 0
cap = cv2.VideoCapture(0)

detector = HD.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    LmksList = detector.findPosition(img)

    # Print position info of any point of choice
    if LmksList:
        print(LmksList[0])

    # Frame rate calculation
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    # Display Frame Per Second (FPS)
    cv2.putText(img, "FPS: "+
    str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 
                3, (0, 0, 0), 4)

    # Show live detection results
    cv2.imshow("Live Hand Tracking", img)
    cv2.waitKey(1)