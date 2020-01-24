import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# setup initial location of window
# r,h,c,w - region of image
#           simply hardcoded the values
# r, h, c, w = 200, 20, 300, 20

ret,frame = cap.read()
boundingbox = cv2.selectROI(frame)
c, r, w, h = boundingbox[0], boundingbox[1], boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[3]
print(boundingbox)
print(c,r,w,h)
cv2.destroyAllWindows()
track_window = (c, r, w, h)
# set up the ROI for tracking

roi = frame[r:r+h, c:c+w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

pixel = hsv_roi[round((r+h)/2), round((c+w)/2)]

# for pixel in hsv_roi:
#     meanpixel = np.array(pixel[0], pixel[1]

upper = np.array([pixel[0] + 20, pixel[1] + 20, pixel[2] + 30])
lower = np.array([pixel[0] - 20, pixel[1] - 20, pixel[2] - 30])
print(pixel, lower, upper)

mask = cv2.inRange(hsv_roi, lower, upper)
# image_mask = cv2.inRange(hsv_roi, mask[0], mask[1])
cv2.imshow("mask", mask)



roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.imshow("hist", roi_hist)
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 1)

while (1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow('img2', img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break

    else:
        break

cv2.destroyAllWindows()
cap.release()
