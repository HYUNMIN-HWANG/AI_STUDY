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

### npy load
x_data = np.load('../data/LPD_competition/npy/data_x_train4.npy', allow_pickle=True)
x_data = np.resize(x_data, (48000, 100*100*3))
print(x_data.shape)    # (48000, 100, 100, 3)
y_data = np.load('../data/LPD_competition/npy/data_y_train4.npy', allow_pickle=True)
print(y_data.shape)    # (48000,)
x_pred = np.load('../data/LPD_competition/npy/data_x_pred4.npy', allow_pickle=True)
x_pred = np.resize(x_pred, (48000,100*100*3))
print(x_pred.shape)     # (72000, 100, 100, 3)


x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=42)
print(x_train.shape, x_valid.shape)
print(y_train.shape, y_valid.shape)
# (43200, 100, 100, 3) (4800, 100, 100, 3)
# (43200,) (4800,)

x_train = x_train / 255.
x_valid = x_valid / 255.

# train_datagen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=20,
#     fill_mode='nearest'
# )

# test_datagen = ImageDataGenerator()

batch = 16
# train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
# valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch)
# pred_generator = test_datagen.flow(x_pred, shuffle=False)

# kf = KFold(n_splits=5, shuffle=True, random_state=45)

model = XGBClassifier(n_jobs = -1, use_label_encoder=False, learning_rate=0.01, n_estimators=100)
    # tree_method = 'cpu_hist', predictor='cpu_predictor')

model.fit(x_train, y_train, verbose=1, eval_metric='mlogloss', eval_set =[(x_train, y_train), (x_valid, y_valid)], early_stopping_rounds=20)

result = model.score(x_valid, y_valid)
print("model.score : ", result)


y_pred = model.predict(x_pred)
print(y_pred)
print(y_pred.shape)


submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
submission['prediction'] = np.argmax(result, axis = 1)
submission.to_csv('../data/LPD_competition/sub_0319_1.csv',index=True)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >


# acc = accuracy_score(y_valid, y_pred)
# print("acc score : ", acc)