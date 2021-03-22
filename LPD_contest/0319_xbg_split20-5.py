import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50, ResNet101, ResNet50V2
from tensorflow.keras.applications.inception_v3 import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier

start_now = datetime.datetime.now()
result_list = []

### npy load
# x_data = np.load(f'../data/LPD_competition/npy/data_x_120_140.npy', allow_pickle=True)
x_data = np.load(f'../data/LPD_competition/npy/data_x_560_580.npy', allow_pickle=True)
x_data = np.resize(x_data, (960, 100*100*3))
print(x_data.shape)    # (960, 30000)

# y_data = np.load(f'../data/LPD_competition/npy/data_y_120_140.npy', allow_pickle=True)
y_data = np.load(f'../data/LPD_competition/npy/data_y_560_580.npy', allow_pickle=True)
y_data = y_data - 560
print(y_data.shape)    # (960,)
print(y_data)

z = 0
for i in range(4) :
    print(z,">>>>>>>>>>>>>>>>>")

    # 다섯 카테고리씩 다시 확인한다.
    now_start = i * 48 * 5
    now_end = now_start + 48 * 5
    now_x_data = x_data[now_start:now_end]
    now_y_data = y_data[now_start:now_end] - 5*i
    print(now_y_data[:10])
    print(now_y_data[-10:])

    x_train, x_valid, y_train, y_valid = train_test_split(now_x_data, now_y_data, train_size=0.9, shuffle=True, random_state=42)
    print(x_train.shape, x_valid.shape) # (216, 30000) (24, 30000)
    print(y_train.shape, y_valid.shape) # (216,) (24,)

    x_train = x_train / 255.
    x_valid = x_valid / 255.

    model = XGBClassifier(n_jobs = -1, use_label_encoder=False, learning_rate=0.01, n_estimators=500)

    model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss', eval_set =[(x_train, y_train), (x_valid, y_valid)], early_stopping_rounds=20)

    result = model.score(x_valid, y_valid)
    print("model.score : ", result)

    result_list.append(result)

    # y_pred = model.predict(x_pred)
    # print(y_pred[:40])
    # print(y_pred.shape)

    # submission['prediction'] = y_pred
    # submission.to_csv('../data/LPD_competition/sub_0319_1.csv',index=True)
    z += 1


print(result_list)


for n, score in enumerate(result_list) : 
    print(n, ":", score)



end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

'''
[0.9479166666666666, 
0.9270833333333334, 
0.96875, 
0.9895833333333334, 
0.9895833333333334, 
0.9583333333333334, 
0.8854166666666666, * (120 ~ 139)
    [1.0, 
    0.9583333333333334, 
    0.9583333333333334, 
    0.875]          ** (135 ~ 139)
0.9895833333333334, 
0.9583333333333334, 
0.9479166666666666, 
0.9895833333333334, 
0.9583333333333334, 
0.9375, 
0.9895833333333334, 
0.9375, 
0.9583333333333334,
0.96875, 
0.9583333333333334, 
0.9479166333334,
0.9479166666666666, 
0.9895833333333334666666666,
0.96875, 
0.9479166666666666, 
0.9895833333333334, 
0.96875,
0.9270833333333334, 
0.9791666666666666, 
0.9375, 
0.8854166666666666, * (560 ~ 579)
    0 : 1.0
    1 : 1.0
    2 : 1.0
    3 : 0.9583333333333334
0.9479166666666666, 
0.958833333333334, 
0.96875, 
0.9270833333333334, 
0.93333333333334,
0.9791666666666666, 
0.9583333333333334, 
0.9583333333333334, 
0.96875, 
0.96875, 
0.96875, 
0.9895833333333334, 
0.9583333333333334, 
0.9375, 
0.9479166666 
0.96875, 
0.9895833333333334, 
0.95833333333333666666, 
0.96875, 
0.96875, 
0.9375,
0.96875, 
0.9583333333333334, 
0.9270833333333334, 
0.9791666666666666, 
0.9895833333333334, 
0.96875, 
0.9791666666666666, 
0.96875  
0.9791666666666666, 
0.96875]
'''
# time >>  14:12:23.143077