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

###

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.3,
    validation_split = 0.2,
    preprocessing_function= preprocess_input,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    preprocessing_function= preprocess_input,
)

xy_train = train_datagen.flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(100,100),
    batch_size=50000,
    seed=42,
    subset='training',
    class_mode='categorical'
)   # Found 39000 images belonging to 1000 classes.

xy_val = train_datagen.flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(100,100),
    batch_size=50000,
    seed=42,
    subset='validation',
    class_mode='categorical'
)   # Found 9000 images belonging to 1000 classes.  

xy_pred = test_datagen.flow_from_directory(
    '../data/LPD_competition/test_test',
    target_size=(100,100),
    batch_size=72000,
    seed=42,
    class_mode=None
)   # Found 72000 images belonging to 1 classes.

# numpy save

# train
np.save('../data/LPD_competition/npy/data_x_train1.npy', arr=xy_train[0][0], allow_pickle=True)
print('x save')
np.save('../data/LPD_competition/npy/data_y_train1.npy', arr=xy_train[0][1], allow_pickle=True)
print('y save')

x_train = np.load('../data/LPD_competition/npy/data_x_train1.npy', allow_pickle=True)
print(x_train.shape)    # (39000, 100, 100, 3)
y_train = np.load('../data/LPD_competition/npy/data_y_train1.npy', allow_pickle=True)
print(y_train.shape)    # (39000, 1000)

# validation
np.save('../data/LPD_competition/npy/data_x_val1.npy', arr=xy_val[0][0], allow_pickle=True)
print('x save')
np.save('../data/LPD_competition/npy/data_y_val1.npy', arr=xy_val[0][1], allow_pickle=True)
print('y save')

x_val = np.load('../data/LPD_competition/npy/data_x_val1.npy', allow_pickle=True)
print(x_val.shape)  # (9000, 100, 100, 3)
y_val = np.load('../data/LPD_competition/npy/data_y_val1.npy', allow_pickle=True)
print(y_val.shape)  # (9000, 1000)

# # predict
np.save('../data/LPD_competition/npy/data_x_pred1.npy', arr=xy_pred[0], allow_pickle=True)
print('x save')

x_pred = np.load('../data/LPD_competition/npy/data_x_pred1.npy', allow_pickle=True)
print(x_pred.shape) # (72000, 100, 100, 3)


