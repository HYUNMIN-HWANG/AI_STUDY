import cv2 as cv
import numpy as np


img_color = cv.imread('../data/DACON_vision2/dirty_mnist_2nd/00000.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
ret, img_binary = cv.threshold(img_gray, 127, 255, 0)
contours, hierachy = cv.findContours(img_binary, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

for cnt in contours :
    cv.drawContours(img_color, [cnt], 0, (255,0,0),3)


cv.imshow("result_color", img_color)
# cv.waitKey(0)

for cnt in contours :
    area = cv.contourArea(cnt)
    print(area)

cv.imshow("result_color2", img_color)
cv.waitKey(0)