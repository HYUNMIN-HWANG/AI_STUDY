import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionResNetV2, EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Embedding, Conv1D, Dropout, MaxPool1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import warnings
warnings.filterwarnings('ignore')

########### npy save

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    rotation_range=5,
    zoom_range=5,
    shear_range=0.5,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
'''
xy_data = train_datagen.flow_from_directory(
    '../data/LPD_competition/train',
    target_size=(80,80),
    batch_size=50000,
    # class_mode='categorical'
    class_mode='sparse'
)   # Found 48000 images belonging to 1000 classes.

print(xy_data)  # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000299B4548550>
print(xy_data[0][0].shape)  # x > (48000, 80, 80, 3)
print(xy_data[0][1].shape)  # y > (48000,)

x = xy_data[0][0].reshape(48000, 80*80, 3)
y = xy_data[0][1]

np.save('../data/LPD_competition/npy/data_x2_2d.npy', arr=x)
np.save('../data/LPD_competition/npy/data_y2_2d.npy', arr=y)
'''

########### npy load

#1. DATA
x_data = np.load('../data/LPD_competition/npy/data_x2_2d.npy')
y_data = np.load('../data/LPD_competition/npy/data_y2_2d.npy')
print(x_data.shape, y_data.shape)   # (48000, 6400, 3) (48000,)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=42)
print(x_train.shape, x_test.shape, x_val.shape)  # (30720, 6400, 3) (9600, 6400, 3) (7680, 6400, 3)
print(y_train.shape, y_test.shape, y_val.shape)  # (30720,) (9600,) (7680,)

#2. Modeling
def my_model() :
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=64, input_length=6400*3)) 
    model.add(Conv1D(1024, 2, padding='same', activation='relu')) 
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1000, activation='softmax'))
    return model

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4)
path = '../data/LPD_competition/cp/cp_0317_1_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

#2. Modeling
model = my_model()
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.01), metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_val, y_val), callbacks=[es, lr, cp])

result = model.evaluate(x_test, y_test, batch_size=16)
print("loss ", result[0])
print("acc ", result[1])

'''


#4. Predict
submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)


test_image = glob('../data/LPD_competition/test/*.jpg')
# print(len(test_image))      # 72000

model = load_model('../data/LPD_competition/cp/cp_0317_1_6.9406.hdf5')  # all 718

# model.summary()

for img in test_image :
    now_img = os.path.basename(img)
    print("\n", now_img)

    copy_img = img 
    img1 = load_img(copy_img , color_mode='rgb', target_size=(80,80)) 
    img1 = img1.resize((80,80))
    img1 = np.array(img1)/255.
    img1 = img1.reshape(-1, 6400,3)
    # print(img1.shape)   # (1, 128, 128, 3)
    pred = np.argmax(model.predict(img1))

    print(pred)

    submission.loc[now_img,:] = pred
    # print(submission.head())

submission.to_csv('../data/LPD_competition/sub_0317_1.csv', index=True)
'''