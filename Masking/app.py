import cv2
import numpy as np
import argparse

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

parser = argparse.ArgumentParser()
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = cv2.VideoCapture(args.image)

while True:

    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low_green, high_green)
    
    cv2.imshow('frame', mask)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()