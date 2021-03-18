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
import datetime 

start_now = datetime.datetime.now()


########### npy load

x_train = np.load('../data/LPD_competition/npy/data_x_train2.npy', allow_pickle=True)
x_train = x_train.reshape(-1, 100*100, 3)
print(x_train.shape)    # (39000, 100*100, 3)
y_train = np.load('../data/LPD_competition/npy/data_y_train2.npy', allow_pickle=True)
print(y_train.shape)    # (39000, )

x_val = np.load('../data/LPD_competition/npy/data_x_val2.npy', allow_pickle=True)
x_val = x_val.reshape(-1, 100*100, 3)
print(x_val.shape)  # (9000, 100*100, 3)
y_val = np.load('../data/LPD_competition/npy/data_y_val2.npy', allow_pickle=True)
print(y_val.shape)  # (9000, )

x_pred = np.load('../data/LPD_competition/npy/data_x_pred2.npy', allow_pickle=True)
x_pred = x_pred.reshape(-1, 100*100, 3)
print(x_pred.shape) # (72000, 100*100, 3)

#2. Modeling
def my_model() :
    model = Sequential()
    model.add(Embedding(input_dim=1000, output_dim=64, input_length=100*100*3)) 
    model.add(Dropout(0.3))
    model.add(Conv1D(1024, 2, padding='same', activation='relu')) 
    model.add(MaxPool1D(2, padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1000, activation='softmax'))
    return model

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4)
path = '../data/LPD_competition/cp/cp_0318_3_embedding.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

#2. Modeling
model = my_model()
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.01), metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer=Adam(0.01), metrics=['acc'])
model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_val, y_val), callbacks=[es, lr, cp])

result = model.evaluate(x_val, y_val, batch_size=16)
print("loss ", result[0])
print("acc ", result[1])



#4. Predict
submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

model = load_model('../data/LPD_competition/cp/cp_0318_3_embedding.hdf5')

print("predict >>>>>>>>>>>>>> ")

result = model.predict(x_pred, verbose=True)
print(result.shape)
submission['prediction'] = np.argmax(result, axis = 1)
submission.to_csv('../data/LPD_competition/sub_0318_3.csv',index=True)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

# score
