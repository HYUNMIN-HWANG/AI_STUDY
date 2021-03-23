# y label 다시 하기

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50, ResNet101, EfficientNetB2
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.resnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from PIL import Image
from sklearn.linear_model import LinearRegression

start_now = datetime.datetime.now()

y_data = np.load('../data/LPD_competition/npy/crop_y_train1.npy', allow_pickle=True)
print(y_data.shape)    # (48090,)
print(len(y_data))     # 48090

# 0부터 120까지 라벨링
for i in range (121) : 
    start = i * 48
    end = start + 48
    y_data[start:end] = i

print(y_data[36:50])        # [0 0 0 0 0 0 0 0 0 0 0 0 1 1]
print(y_data[5790:5809])    # [ 120  120  120  120  120  120  120  120  120  120  120  120  120  120 120  120  120  120 5808]

# 121 라벨링 (이미지 55개)
y_data[5808:5808+55] = 121
print(len(y_data[5808:5808+55]))    # 55
print(y_data[5808-1:5808+57])

# 122 라벨링 (이미지 57개)
y_data[5863:5863+57] = 122
print(y_data[5863-1:5863+57+1])

# 123 라벨링 (이미지 57개)
y_data[5920:5920+57] = 123
print(y_data[5920:5920+57])

# 124 라벨링 (이미지 53개)
y_data[5977:5977+53] = 124
print(y_data[5977:5977+53])

# 125 라벨링 (이미지 56개)
y_data[6030:6030+56] = 125
print(y_data[6030:6030+56])

# 126 라벨링 (이미지 55개)
y_data[6086:6086+55] = 126

# 127 라벨링 (이미지 50개)
y_data[6141:6141+50] = 127

# 128 ~ 131 라벨링
j = 0
for i in range(128 , 132) :
    term = 48 * j
    start = 6191 + term
    y_data[start : start + 48] = i 

    j += 1
print(len(y_data[6191:6383]))

# 132 라벨링 (이미지 54개)
y_data[6383:6383+54] = 132

# 133 라벨링 (이미지 53개)
y_data[6437:6437+53] = 133

# 134
y_data[6490:6490+48] = 134

# 135
y_data[6538:6538+48] = 135

# 136 (이미지 54개)
y_data[6586:6586+54] = 136

# 137
y_data[6640:6640+48] = 137

# 138
y_data[6688:6688+48] = 138

# 139 (이미지 53개)
y_data[6736:6736+53] = 139  
print(len(y_data[6736:6736+53]))    # 53

# 140 ~ 574 라벨링 (6789 : 27669)
j = 0
for i in range(140, 575) :
    term = 48 * j
    start = 6789 + term
    print(start)    # 27621
    y_data[start : start + 48] = i 

    j += 1

print(len(y_data[6789:27621+48]))  # 20880 = 48 * 435 개 라벨링
print(y_data[6789-1:6789+50])
print(y_data[27669-5:27669+1])

# 575 (이미지 57개)
y_data[27669:27669+57] = 575  

# 576
y_data[27726:27726+48] = 576

# 577 (이미지 52개)
y_data[27774:27774+52] = 577

# 578 (이미지 52개)
y_data[27826:27826+52] = 578

# 579 (이미지 52개)
y_data[27878:27878+52] = 579

# 580 ~ 999 라벨링 (27930 ~ 48090)
j = 0
for i in range(580, 1000) :
    term = 48 * j
    start = 27930 + term
    print(start)    # 48042
    y_data[start : start + 48] = i 

    j += 1
print(len(y_data[27930:48090]))  # 20160 = 48 * 420 개 라벨링
print(y_data[27930-1:27930+50])
print(y_data[-5:])

print(y_data)

y_data = to_categorical(y_data)
print(y_data.shape) # (48090, 1000)


np.save('../data/LPD_competition/npy/crop_y_train1_label.npy', arr=y_data, allow_pickle=True)


