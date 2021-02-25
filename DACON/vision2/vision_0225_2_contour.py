import cv2 as cv
import numpy as np


img_color = cv.imread('../data/DACON_vision2/dirty_mnist_2nd/00000.png')
img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(img_gray, (3,3), cv.BORDER_DEFAULT)

ret, img_binary = cv.threshold(blur, 170, 255, cv.THRESH_BINARY)
cv.imshow("img_binary", img_binary)
# cv.waitKey(0)

kernel = np.ones((3,3), np.uint8)
erode = cv.erode(img_binary, kernel, iterations = 1)
# cv.imshow("erode", erode)
# cv.waitKey(0)

# morph = cv.morphologyEx(erode, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)), iterations=2)
morph = cv.morphologyEx(erode, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2,2)), iterations=2)
cv.imshow("morph", morph)
# cv.waitKey(0)

# 외곽선 검출
# contours , hierarchy = cv.findContours(morph , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
contours , hierarchy = cv.findContours(morph , cv.RETR_EXTERNAL , cv.CHAIN_APPROX_NONE)


color = cv.cvtColor(morph, cv.COLOR_GRAY2BGR)           # 이진화 이미지를 color이미지로 복사(확인용)
cv.drawContours(color , contours , -1 , (0,255,0),3)    # 초록색으로 외곽선을 그려준다.

bR_arr = []
digit_arr = []
digit_arr2 = []
count = 0

for i in range(len(contours)) :
    bin_tmp = morph.copy()
    x,y,w,h = cv.boundingRect(contours[i])
    bR_arr.append([x,y,w,h])

#x값을 기준으로 배열을 정렬
bR_arr = sorted(bR_arr, key=lambda num : num[0], reverse = False)
print(bR_arr[:5])
print(len(bR_arr))  # 17


#작은 노이즈데이터 버림,사각형그리기
for x,y,w,h in bR_arr :
    tmp_y = bin_tmp[y-2:y+h+2,x-2:x+w+2].shape[0]
    tmp_x = bin_tmp[y-2:y+h+2,x-2:x+w+2].shape[1]
    if  tmp_x and tmp_y > 15 :
        count += 1 
        cv.rectangle(color,(x-2,y-2),(x+w+2,y+h+2),(0,0,255),1)
        digit_arr.append(bin_tmp[y-2:y+h+2,x-2:x+w+2])
        # print(bin_tmp[y-2:y+h+2,x-2:x+w+2])
digit_arr2.append(digit_arr)
print(len(digit_arr))        # 13
print(len(digit_arr2))       # 1
# cv.imshow('contours',color)
# k = cv.waitKey(0)
# cv.destroyAllWindows()

print(digit_arr[0][0])
print(digit_arr2[0][3])
# a = np.array(digit_arr2[0][3])
# print(a.shape)

#리스트에 저장된 이미지를 32x32의 크기로 리사이즈해서 순서대로 저장
for i in range(0,len(digit_arr2)) :
    print("i ", i)  # 사각형으로 자른 이미지 하나 하나
    for j in range(len(digit_arr2[i])) :
        count += 1 
        print("j ", j)
        if i == 0 :         #1일 경우 비율 유지를 위해 마스크를 만들어 그 위에 얹어줌
            width = digit_arr2[i][j].shape[1]
            print(width)    # 52
            height = digit_arr2[i][j].shape[0]
            print(height)   # 54
            if width < height :
                mask = np.zeros((height,height))
                tmp = (height - width)/2
                mask[0:height, int(tmp):int(tmp)+width] = digit_arr2[i][j]
                digit_arr2[i][j] = cv.resize(mask,(32,32))
            else :
                mask = np.zeros((width, width))
                tmp = (width - height)/2
                mask[int(tmp):int(tmp)+height, 0:width] = digit_arr2[i][j]
                digit_arr2[i][j] = cv.resize(mask,(32,32))
        else:
            digit_arr2[i][j] = cv.resize(digit_arr2[i][j],(32,32))
        if i == 9 : i = -1
        cv.imwrite('../data/DACON_vision2/contour/'+str(i+1)+'_'+str(j)+'.png',digit_arr2[i][j])

'''
cv.imshow("result_color", img_color)
# cv.waitKey(0)

for cnt in contours :
    area = cv.contourArea(cnt)
    print(area)

cv.imshow("result_color2", img_color)
cv.waitKey(0)
'''