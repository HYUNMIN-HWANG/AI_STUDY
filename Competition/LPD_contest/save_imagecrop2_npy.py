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

### train data

x_data = np.load('../data/LPD_competition/npy/data_x_train5.npy', allow_pickle=True)
print(x_data.shape)    # (48090, 200, 200, 3)

# y_data = np.load('../data/LPD_competition/npy/data_y_train5.npy', allow_pickle=True)
# print(y_data.shape)    # (48090,)

# y_data = to_categorical(y_data)
# print(y_data.shape) # (48090, 1000)

sharpen = np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]])

rectangle = (5, 5, 195, 195)
mask = np.zeros((200,200), np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

data = []
label = []

for i in range(48090) :
    print(i)
    one_img = x_data[i] 
    # blur_img = cv.GaussianBlur(one_img, (3,3),0)        # 블러처리
    temp = cv.filter2D(one_img, -1, sharpen)       # 경계선 강조
    # plt.imshow(temp); plt.show()
    cv.grabCut(temp, mask, rectangle, bgdModel, fgdModel, 18, cv.GC_INIT_WITH_RECT) # 배경 제거
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img_rgb_nobg = temp * mask2[:,:,np.newaxis]
    # plt.imshow(img_rgb_nobg); plt.show()

    crop_img = img_rgb_nobg[5:-15, 30:-30].copy()     # 이미지 크롭
    crop_img = cv.resize(crop_img, (100,100), interpolation=cv.INTER_AREA)   # 이미지 작게 리사이즈
    # plt.imshow(crop_img); plt.show()

    # print(one_img.shape)    # (200, 200, 3)
    # print(blur_img.shape)   # (200, 200, 3)
    # print(crop_img.shape)   # (120, 100, 3)

    # plt.imshow(crop_img)
    # plt.show()
    crop_img = np.array(crop_img)
    data.append(crop_img)
    # label.append(i)

data = np.array(data)
# label = np.array(label)

print(data.shape)   # (48090, 120, 100, 3)
# print(label.shape)  # 

np.save('../data/LPD_competition/npy/crop_x_train2.npy', arr=data, allow_pickle=True)
print('x save')
# np.save('../data/LPD_competition/npy/crop_y_train2.npy', arr=label, allow_pickle=True)
# print('y save')

