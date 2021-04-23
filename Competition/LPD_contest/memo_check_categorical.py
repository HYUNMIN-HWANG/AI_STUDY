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


train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split = 0.2,
    # preprocessing_function= preprocess_input,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    # preprocessing_function= preprocess_input,
)

xy_train = train_datagen.flow_from_directory(
    '../LPD_competition/traintrain',
    target_size=(100,100),
    batch_size=50000,
    seed=42,
    subset='training',
    class_mode='categorical'
)   # Found 936 images belonging to 24 classes.

xy_val = train_datagen.flow_from_directory(
    '../LPD_competition/traintrain',
    target_size=(100,100),
    batch_size=50000,
    seed=42,
    subset='validation',
    class_mode='categorical'
)   # Found 216 images belonging to 24 classes.

print(xy_train[0][1][0])   # [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(xy_train[0][1][1])   # [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(xy_train[0][1][2])   # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
print(xy_train[0][1][3])   # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]

