import cv2 
import math
import mediapipe as mp
import numpy as np 

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpdraw = mp.solutions.drawing_utils

# Function to calculate distance between two points in 2D space
def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

cx, cy, cz = 0, 0, 0
def center_point(x1, y1, z1):
    global cx
    cx = x1
    global cy
    cy = y1
    global cz
    cz = z1

# OpenCV code to capture video from the default camera
cap = cv2.VideoCapture(0)
cap.set(3, 720)  # Set the width of the frame
cap.set(4, 480)  # Set the height of the frame
mylmList = []
img_counter=0
# Main loop to process video frames
while True:
    isopen, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR image to RGB for Mediapipe processing
    results = hands.process(img)  # Process the frame with Mediapipe Hands
    allHands = []
    h, w, c = frame.shape  # Get the height, width, and number of channels of the frame
    
    # Process each detected hand in the frame
    if results.multi_hand_landmarks:
        for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
            myHand = {}
            mylmList = []
            xList = []
            yList = []
            zList = []
            
            # Extract landmark points and store them in lists
            for id, lm in enumerate(handLms.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([id, px, py])
                xList.append(px)
                yList.append(py)
                zList.append((pz))
            
            allHands.append(myHand)
            
            if mylmList != 0:
                try:
                    height = zList[0]

                    x1 = int(xList[9])
                    y1 = int(yList[9])
                    z1 = int(zList[9])

                    distance =  math.sqrt((x1 - x1)**2 + (y1 - y1)**2 + (z1 - z1)**2)
                    height = zList[5]

                    print(x1-cx, y1-cy, z1-cz)

                    if x1 < cx and abs(cx-x1) > 50:
                        print("Right")
                    elif x1 > cx and abs(cx - x1) > 50:
                        print("Left")

                    if y1 < cy and abs(cy - y1) > 50:
                        print("Forward")
                    elif y1 > cy and abs(cy - y1) > 50:
                        print("Backward")

                    # if z1 < cz:
                    #     print("Near")
                    # elif z1 > cz:
                    #     print("Far")
                except:
                    pass
    
    # Display the frame with annotations
    cv2.imshow('Hand Distance Measurement', frame)
    
    # Exit the loop if 'q' key is pressed
    k = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff==ord('c'):
        center_point(x1, y1, z1)
        print("Center Point Set")
    elif cv2.waitKey(1) & 0xff==ord('q'):
        break