import numpy as np
import pandas as pd
import cv2 as cv

from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import PIL.Image as pilimg

######################################################
# File Load
train = pd.read_csv('../data/DACON_vision2/dirty_mnist_2nd_answer.csv')
print(train.shape)  # (50000, 27)

sub = pd.read_csv('../data/DACON_vision2/sample_submission.csv')
print(sub.shape)    # (5000, 27)

######################################################

#1. DATA

#### train

df_x = []

for i in range(0,50000):
    if i < 10 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/0000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/000' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/00' + str(i) + '.png'
    elif i >= 1000 and i < 10000 :
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/0' + str(i) + '.png'
    else : 
        file_path = '../data/DACON_vision2/dirty_mnist_2nd/' + str(i) + '.png'
    
    # image = pilimg.open(file_path)
    img = cv.imread(file_path)
    img = cv.resize(img,(80,80))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    threshold, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)

    pix = np.array(thresh)
    pix = pd.DataFrame(pix)
    df_x.append(pix)

x = pd.concat(df_x)
x = x.values
print("x.shape ", x.shape)       # (4000000, 80)
print(x[0,:])
x = x.reshape(50000, 80,80, 1)

y = train.iloc[:,1:]
print("y.shape ", y.shape)    # (50000, 26)

np.save('../data/DACON_vision2/npy/vision_x6.npy', arr=x)
np.save('../data/DACON_vision2/npy/vision_y6.npy', arr=y)


#### pred
df_pred = []

for i in range(0,5000):
    if i < 10 :
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/5000' + str(i) + '.png'
    elif i >=10 and i < 100 :
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/500' + str(i) + '.png'
    elif i >= 100 and i <1000 :
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/50' + str(i) + '.png'
    else : 
        file_path = '../data/DACON_vision2/test_dirty_mnist_2nd/5' + str(i) + '.png'
    # image = pilimg.open(file_path)
    img = cv.imread(file_path)
    img = cv.resize(img,(80,80))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
    threshold, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY)
    pix = np.array(thresh)
    pix = pd.DataFrame(pix)
    df_pred.append(pix)

x_pred = pd.concat(df_pred)
x_pred = x_pred.values
print(x_pred.shape)       #  (400000, 80)
x_pred = x_pred.reshape(5000, 80,80, 1)

np.save('../data/DACON_vision2/npy/vision_x_pred6.npy', arr=x_pred)


# 거진 26시간 돌아감
# t까지 훈련해서 제출했는데 스코어 0.5194923077	
# 이거 아니다.