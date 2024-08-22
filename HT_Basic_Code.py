# To set up the environment, in your terminal (Windows), do: 
# pip install opencv-python
# pip install mediapipe
import cv2
import mediapipe as mp
import time

# Video Capture system setup
cap = cv2.VideoCapture(0)

# Necessary detection and display functions
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Frame rate calculation variables
prevTime = 0
curTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)
    # Return hands information if there are hands being detected
    # Otherwise returns None

    # If there are hands being detected
    if(results.multi_hand_landmarks): 

        # Iterate through the hands (>=1)
        for handLmks in results.multi_hand_landmarks:

            # get id and landmark info of each hand
            for id, lm in enumerate(handLmks.landmark):
                # id: from 0 to 19 inclusively
                # lm: include x, y, z coordinates

                height, width, channel = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)

                # label each point with corresponding numbers (optional)
                    # cv2.putText(img, str(id), (cx, cy),
                    # cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

            # Draw points
            mpDraw.draw_landmarks(img, handLmks, mpHands.HAND_CONNECTIONS)

    # Frame rate calculation
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    # Display Frame Per Second (FPS)
    cv2.putText(img, "FPS: "+str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 
                3, (0, 0, 0), 4)

    # Show live detection results
    cv2.imshow("Live Hand Tracking", img)
    cv2.waitKey(1)