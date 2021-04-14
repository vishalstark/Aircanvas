import numpy as np
import cv2
from collections import deque

blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

skinLower = np.array([108, 23, 82], dtype=np.uint8)
skinUpper = np.array([179, 255, 255], dtype=np.uint8)
        
kernel = np.ones((5, 5), np.uint8)

redpoints = [deque(maxlen=512)]
greenpoints = [deque(maxlen=512)]
bluepoints = [deque(maxlen=512)]
blackpoints = [deque(maxlen=512)]

blueindex = 0
greenindex = 0
redindex = 0
blackindex = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
colourIndex = 0

paintWindow = np.zeros((471,636,3)) + 255

def frame_window(frame):
    frame = cv2.rectangle(frame,(0,1),(140,100),(0,255,0),1)    
    frame = cv2.rectangle(frame, (160,1), (255,65), colors[0], -1)
    frame = cv2.rectangle(frame, (275,1), (370,65), colors[1], -1)
    frame = cv2.rectangle(frame, (390,1), (485,65), colors[2], -1)
    frame = cv2.rectangle(frame, (505,1), (600,65), colors[3], -1)
    cv2.putText(frame, "CLEAR ALL", (30, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLACK", (520, 33), cv2.FONT_HERSHEY_PLAIN, 1, (150,150,150), 2, cv2.LINE_AA)

camera = cv2.VideoCapture(0)

def roi_preprocess(frame):

    blur = cv2.GaussianBlur(frame, (3,3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(frame, skinLower, skinUpper)
    blur = cv2.medianBlur(mask, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    new_frame = cv2.dilate(blur, kernel)

    return new_frame

while True:

    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    roi=frame[1:100, 0:140]
    mask = roi_preprocess(roi)
    noWhitePixel = np.sum(roi == 255)
       
    frame_window(frame)

    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    (_, cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    center = None


    if noWhitePixel>100:
        redpoints = [deque(maxlen=512)]
        greenpoints = [deque(maxlen=512)]
        bluepoints = [deque(maxlen=512)]
        blackpoints = [deque(maxlen=512)]

        redindex = 0
        greenindex = 0
        blueindex = 0
        blackindex = 0

        paintWindow[67:,:,:] = 255


    if len(cnts) > 0:
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))


        if center[1] <= 65:
            if 160 <= center[0] <= 255:
                    colourIndex = 0 
            elif 275 <= center[0] <= 370:
                    colourIndex = 1
            elif 390 <= center[0] <= 485:
                    colourIndex = 2 
            elif 505 <= center[0] <= 600:
                    colourIndex = 3
        else :
            if colourIndex == 0:
                bluepoints[blueindex].appendleft(center)
            elif colourIndex == 1:
                greenpoints[greenindex].appendleft(center)
            elif colourIndex == 2:
                redpoints[redindex].appendleft(center)
            elif colourIndex == 3:
                blackpoints[blackindex].appendleft(center)

    else:
        redpoints.append(deque(maxlen=512))
        redindex += 1
        greenpoints.append(deque(maxlen=512))
        greenindex += 1
        bluepoints.append(deque(maxlen=512))
        blueindex += 1
        blackpoints.append(deque(maxlen=512))
        blackindex += 1

    points = [bluepoints, greenpoints, redpoints, blackpoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
