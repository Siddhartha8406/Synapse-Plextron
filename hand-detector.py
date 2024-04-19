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
            
            # # Calculate bounding box around the hand landmarks
            # xmin, xmax = min(xList), max(xList)
            # ymin, ymax = min(yList), max(yList)
            # boxW, boxH = xmax - xmin, ymax - ymin
            # bbox = xmin, ymin, boxW, boxH
            # cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
            
            # # Store hand information in a dictionary
            # myHand["lmList"] = mylmList
            # myHand["bbox"] = bbox
            # myHand["center"] = (cx, cy)
            # myHand["type"] = handType.classification[0].label
            #if you dont flip the image
            ''' if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
            else:
                        myHand["type"] = "Right"'''
            allHands.append(myHand)
            
            # Draw landmarks and bounding box on the frame
            # mpdraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            # cv2.rectangle(frame, (bbox[0] - 20, bbox[1] - 20), (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20), (255, 0, 255), 2)
            # cv2.putText(frame, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            
            # Calculate and display the distance between two specific landmarks of the hand
            if mylmList != 0:
                try:
                    x, y = mylmList[5][1], mylmList[5][2]
                    x2, y2 = mylmList[17][1], mylmList[17][2]
                    height = zList[0]

                    x1, y1, z1 = xList[4], yList[4], zList[4]
                    x2, y2, z2 = xList[8], yList[8], zList[8]

                    # print(f"Distance:", (math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)))

                    distance =  math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
                    height = zList[5]

                    print(distance, height, sep="   |   ")

                    # cv2.rectangle(frame, (xmax - 80, ymin - 80), (xmax + 20, ymin - 20), (255, 0, 255), cv2.FILLED)
                    # cv2.putText(frame, f"{int(height)}cm", (xmax - 80, ymin - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
          
                except:
                    pass
    
    # Display the frame with annotations
    cv2.imshow('Hand Distance Measurement', frame)
    
    # Exit the loop if 'q' key is pressed
    k = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break