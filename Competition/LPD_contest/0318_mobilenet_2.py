import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.inception_v3 import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
import datetime 

start_now = datetime.datetime.now()

#1. DATA
### npy load
x_train = np.load('../data/LPD_competition/npy/data_x_train2.npy', allow_pickle=True)
print(x_train.shape)    # (39000, 100, 100, 3)
y_train = np.load('../data/LPD_competition/npy/data_y_train2.npy', allow_pickle=True)
print(y_train.shape)    # (39000, )

x_val = np.load('../data/LPD_competition/npy/data_x_val2.npy', allow_pickle=True)
print(x_val.shape)  # (9000, 100, 100, 3)
y_val = np.load('../data/LPD_competition/npy/data_y_val2.npy', allow_pickle=True)
print(y_val.shape)  # (9000, )

x_pred = np.load('../data/LPD_competition/npy/data_x_pred2.npy', allow_pickle=True)
print(x_pred.shape) # (72000, 100, 100, 3)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)
x_pred = preprocess_input(x_pred)

'''
#2. Modeling
mobile_net = MobileNet(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
for layer in mobile_net.layers:
        layer.trainable = True

top_model = mobile_net.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Flatten()(top_model)
top_model = Dense(1024, activation="relu")(top_model)
top_model = Dropout(0.2) (top_model)
top_model = Dense(1000, activation="softmax")(top_model)
model = Model(inputs=mobile_net.input, outputs = top_model)
model.summary()

#3. Compile, Train, Evaluate
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.06)
path = '../data/LPD_competition/cp/cp_0318_2_mobile.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

batch = 16

model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['acc'])
model.fit(x_train, y_train, epochs=500, batch_size=batch ,
    validation_data=(x_val, y_val), callbacks=[es, lr, cp])

result = model.evaluate(x_val, y_val, batch_size=batch)
print("loss ", result[0])
print("acc ", result[1])

# loss  0.02675320766866207
# acc  0.9941111207008362

'''

#4. Predict
submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

model = load_model('../data/LPD_competition/cp/cp_0318_2_mobile.hdf5')
result = model.evaluate(x_val, y_val, batch_size=16)
print("loss ", result[0])
print("acc ", result[1])

# loss  0.02401334047317505
# acc  0.0010000000474974513

print("predict >>>>>>>>>>>>>> ")
'''
result = model.predict(x_pred, verbose=True)
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))

submission['prediction'] = np.argmax(result, axis = 1)
submission.to_csv('../data/LPD_competition/sub_0318_2.csv',index=True)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >  0:51:44.348498

# score 0.111
'''