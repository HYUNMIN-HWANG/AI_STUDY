# C:\lotte\lotte\preprocessing\gausian_blur.py
# 를 위한 ~ 이미지 선명하게!

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import PIL.Image as pilimg

# ---------------------------------------------------------------------
# train 데이터 불러오기
train = list()
label = list()
number1 = 1000
number2 = 48

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# 이미지 저장하면서 라벨까지 저장
# train
# train - label 
for aa in range(number1):
    for a in range(number2):
        temp = cv2.imread('C:/lotte_data/LPD_competition/train/' + str(aa)+ '/' + str(a) + '.jpg')
        temp = cv2.resize(temp, (128, 128))
        temp = cv2.filter2D(temp, -1, kernel)
        plt.imshow(temp); plt.show()
        temp = np.array(temp)
        train.append(temp)


# np.array로 바꿔서 쉐잎 확인
filter2D_train_data_100 = np.array(train)
print(filter2D_train_data_100.shape)

# # ---------------------------------------------------------------------
# # npy로 저장

np.save('D:/lotte_data/npy/filter2D_train_data_100.npy', arr = filter2D_train_data_100)
print('===== done =====')