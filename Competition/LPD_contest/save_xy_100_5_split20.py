import numpy as np
from numpy import asarray
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB7, MobileNet
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.utils import to_categorical


### train data


for time in range(50) : 
    data = []
    label = []

    start = time * 20
    end = start + 20 
    print(f"======<< {time} time >>======")
    for i in range(start , end) :
        # print(i)
        img = glob(f'../data/LPD_competition/train/{i}/*.jpg')
        for j in img :
            print(j)
            one_img = Image.open(j)
            one_img = one_img.convert("RGB")
            one_img = one_img.resize((100, 100))
            one_img = np.asarray(one_img)
            data.append(one_img)
            label.append(i)
    
    data = np.array(data)
    label = np.array(label)

    # label = to_categorical(label)
    print(data.shape)   # (960, 100, 100, 3)
    print(label.shape)  # (960, )

    np.save(f'../data/LPD_competition/npy/data_x_{start}_{end}.npy', arr=data, allow_pickle=True)
    print('train save')
    np.save(f'../data/LPD_competition/npy/data_y_{start}_{end}.npy', arr=label, allow_pickle=True)
    print('valid save')

'''
### test data
pred = []
for i in range (72000) : 
    file = f'../data/LPD_competition/test_test/test/{i}.jpg'
    print(file)
    one_img = Image.open(file)
    one_img = one_img.convert("RGB")
    one_img = one_img.resize((100, 100))
    one_img = np.asarray(one_img)
    pred.append(one_img)
pred = np.array(pred)
print(pred.shape)   # (72000, 100, 100, 3)
# pred_generator = test_datagen.flow(pred, shuffle=False)
np.save('../data/LPD_competition/npy/data_pred_gen1.npy', arr=pred, allow_pickle=True)
print('pred save')
'''