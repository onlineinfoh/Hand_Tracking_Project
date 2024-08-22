import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        # Constructor: set variables 
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # Necessary detection and display functions
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # If there are hands being detected
        if self.results.multi_hand_landmarks: 

            # Iterate through the hands (>=1)
            for handLmks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, 
                    handLmks, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, handNumber = 0, draw = True):

        LmksList = []

        if self.results.multi_hand_landmarks: 
            
            hand = self.results.multi_hand_landmarks[handNumber]

            for id, lm in enumerate(hand.landmark):
                    height, width, channel = img.shape
                    cx, cy = int(lm.x*width), int(lm.y*height)

                    LmksList.append([id, cx, cy])

                    #if draw:
                    #    cv2.circle(img, (cx, cy), 15, (0,0,0), cv2.FILLED)

        return LmksList