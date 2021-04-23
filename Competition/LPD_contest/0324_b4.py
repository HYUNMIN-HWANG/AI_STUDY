import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB4, InceptionV3, MobileNet, ResNet50, ResNet101, Xception
from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.resnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 

submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

start_now = datetime.datetime.now()

### npy load
x_data = np.load('../data/LPD_competition/npy/data_x_train4.npy', allow_pickle=True)
print(x_data.shape)    # (48000, 100, 100, 3)
y_data = np.load('../data/LPD_competition/npy/data_y_train4.npy', allow_pickle=True)
print(y_data.shape)    # (48000,)
x_pred = np.load('../data/LPD_competition/npy/data_x_pred4.npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 100, 100, 3)


#1. DATA
# preprocess
x_data = preprocess_input(x_data)
x_pred = preprocess_input(x_pred)

y_data = to_categorical(y_data)

train_datagen = ImageDataGenerator(
    width_shift_range=(-1,1),
    height_shift_range=(-1,1),
    rotation_range=10,
    zoom_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=42)
print(x_train.shape, x_valid.shape)  # (43200, 100, 100, 3) (4800, 100, 100, 3)
print(y_train.shape, y_valid.shape)  # (43200, 1000) (4800, 1000)


def my_model () :
    transfer = EfficientNetB4(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
    transfer.trainable = True
    top_model = transfer.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Flatten()(top_model)
    top_model = Dense(4048, activation="swish")(top_model)
    top_model = Dropout(0.3) (top_model)
    top_model = Dense(1000, activation="softmax")(top_model)

    model = Model(inputs=transfer.input, outputs = top_model)
    return model

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.06)
path = '../data/LPD_competition/cp/cp_0324_4_b4.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch)
pred_generator = x_pred

model = my_model()
# model.summary()

#3. Compile, Train, Evaluate
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

hist = model.fit_generator(train_generator, epochs=200, steps_per_epoch = len(x_train) // batch ,
    validation_data=valid_generator, validation_steps=10 , callbacks=[es, lr, cp])

model.save_weights('../data/LPD_competition/cp//cp_0324_4_b4_weights.h5')

result = model.evaluate(valid_generator, batch_size=batch)
print("loss ", result[0])   
print("acc ", result[1])  
# EarlyStopping 48
# loss  0.008243024349212646
# acc  0.9978125095367432

#4. Predict
model = load_model('../data/LPD_competition/cp//cp_0324_4_b4.hdf5')
# model.load_weights('../data/LPD_competition/cp//cp_0324_3_b2_weights.h5')

print(">>>>>>>>>>>>>>>> predict >>>>>>>>>>>>>> ")
# tta 나만 터져?

result = model.predict(pred_generator, verbose=True)

# save
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))
result_arg = np.argmax(result, axis = 1)

submission['prediction'] = result_arg
submission.to_csv('../data/LPD_competition/sub_0324_5.csv', index=True)

end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # ime >>  2:42:39.060425 


# score 75.496