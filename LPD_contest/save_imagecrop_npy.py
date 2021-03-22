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

x_data = np.load('../LPD_competition/data_x_train5.npy', allow_pickle=True)
print(x_data.shape)    # (48000, 100, 100, 3)

y_data = np.load('../LPD_competition/data_y_train5.npy', allow_pickle=True)
print(y_data.shape)    # (48000,)

    
sharpen = np.array([[0,-1,0],
                    [-1,5,-1],
                    [0,-1,0]])

rectangle = (5, 5, 95, 95)
mask = np.zeros((200,200), np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

data = []
label = []

for i in range(48000) :
    one_img = x_data[i] 
    blur_img = cv.GaussianBlur(one_img, (3,3),0)       # 블러처리
    blur_img = cv.filter2D(blur_img, -1, sharpen)       # 경계선 강조

    cv.grabCut(blur_img, mask, rectangle, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT) # 배경 제거
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img_rgb_nobg = blur_img * mask2[:,:,np.newaxis]

    crop_img = img_rgb_nobg[20:-20, 30:-30].copy()     # 이미지 크롭
    crop_img = cv.resize(crop_img, (100,100), interpolation=cv.INTER_AREA)   # 이미지 작게 리사이즈

    print(one_img.shape)    # (100, 100, 3)
    print(blur.shape)       # (100, 100, 3)
    print(crop_img.shape)   # (80, 80, 3)

    plt.imshow(crop_img)
    plt.show()

    data.append(crop_img)
    label.append(i)
    
data = np.array(data)
label = np.array(label)

print(data.shape)   # 
print(label.shape)  # 

'''
np.save('../data/LPD_competition/npy/crop_x_train1.npy', arr=data, allow_pickle=True)
print('x save')
np.save('../data/LPD_competition/npy/crop_y_train1.npy', arr=label, allow_pickle=True)
print('y save')
'''

'''

### test data

x_pred = np.load('../LPD_competition/data_x_pred5.npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 12288)



pred = []
for m in range(72000) : 
    file = f'../data/LPD_competition/test_test/test/{m}.jpg'
    print(file)
    one_img = cv.imread(file, cv.IMREAD_COLOR)          # 이미지 불러오기 
    one_img = cv.cvtColor(one_img,cv.COLOR_BGR2RGB)     # RGB로 바꾸기
    crop_img = one_img[10:-10, 50:-50].copy()           # 이미지 크롭
    crop_img = cv.GaussianBlur(crop_img, (3,3),0)       # 블러처리
    crop_img = cv.filter2D(crop_img, -1, sharpen)       # 경계선 강조

    cv.grabCut(crop_img, mask, rectangle, bgdModel, fgdModel, 20, cv.GC_INIT_WITH_RECT) # 배경 제거
    mask2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
    img_rgb_nobg = crop_img * mask2[:,:,np.newaxis]

    crop_img2 = img_rgb_nobg[10:-10, 20:-20].copy()     # 이미지 크롭
    crop_img2 = cv.resize(crop_img2, (100, 120), interpolation=cv.INTER_AREA)   # 이미지 작게 리사이즈
    print(crop_img2.shape)
    pred.append(crop_img2)

pred = np.array(pred)
print(pred.shape)

np.save('../data/LPD_competition/npy/crop_x_pred1.npy', arr=pred, allow_pickle=True)
print('predict save')

'''