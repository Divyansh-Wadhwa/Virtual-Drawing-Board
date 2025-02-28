import cv2 
import mediapipe as mp 
import os
import time 
import numpy as np
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 1000)  # Set height
cap.set(10, 150)  # Set brightness
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

pasttime = 0

folder = 'colors'
mylist = os.listdir(folder)
overlist = []
col = [0, 0, 255]  # Default color (red)

for i in mylist:
    image = cv2.imread(f'{folder}/{i}')
    print(image.shape)
    overlist.append(image)

header = overlist[0]

print(mylist)
xp, yp = 0, 0
canvas = np.zeros((480, 640, 3), np.uint8)

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img)
    lanmark = []

    if results.multi_hand_landmarks:
        for hn in results.multi_hand_landmarks:
            for id, lm in enumerate(hn.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lanmark.append([id, cx, cy])
            mpdraw.draw_landmarks(frame, hn, mpHands.HAND_CONNECTIONS)
    
    if len(lanmark) != 0:
        x1, y1 = lanmark[8][1], lanmark[8][2]
        x2, y2 = lanmark[12][1], lanmark[12][2]

        if lanmark[8][2] < lanmark[6][2] and lanmark[12][2] < lanmark[10][2]:
            xp, yp = 0, 0
            print('Selection mode')

            #
            if y1 < 100:
                if 71 < x1 < 142:
                    header = overlist[7]
                    col = (0, 0, 0)
                if 142 < x1 < 213:
                    header = overlist[6]
                    col = (226, 43, 138)
               
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, cv2.FILLED)

        elif lanmark[8][2] < lanmark[6][2]:
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if col == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), col, 100, cv2.FILLED)
                cv2.line(canvas, (xp, yp), (x1, y1), col, 100, cv2.FILLED)
            cv2.line(frame, (xp, yp), (x1, y1), col, 25, cv2.FILLED)
            cv2.line(canvas, (xp, yp), (x1, y1), col, 25, cv2.FILLED)
            print('Drawing mode')
            xp, yp = x1, y1

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    frame[0:100, 0:640] = header


    ctime = time.time()
    fps = 1 / (ctime - pasttime)
    pasttime = ctime
    cv2.putText(frame, f'FPS: {int(fps)}', (490, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

    cv2.imshow('cam', frame)
    cv2.imshow('canvas', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
