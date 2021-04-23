import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50, ResNet101, Xception
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.resnet import preprocess_input
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
x_data = np.load('../data/LPD_competition/npy/crop_x_train1.npy', allow_pickle=True)
print(x_data.shape)    # (48090, 100, 100, 3)
y_data = np.load('../data/LPD_competition/npy/crop_y_train1_label.npy', allow_pickle=True)
print(y_data.shape)    # (48090, 1000)
x_pred = np.load('../data/LPD_competition/npy/data_x_pred4.npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 100, 100, 3)


#1. DATA
# preprocess
x_data = preprocess_input(x_data)
x_pred = preprocess_input(x_pred)


train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    width_shift_range= 0.05,
    height_shift_range= 0.05
)

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size=0.9, shuffle=True, random_state=42)
print(x_train.shape, x_valid.shape)  # (43281, 100, 100, 3) (4809, 100, 100, 3)
print(y_train.shape, y_valid.shape)  # (43281, 1000) (4809, 1000)

dropout_rate = 0.2

def my_model () :
    transfer = Xception(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
    for layer in transfer.layers:
            layer.trainable = True
    top_model = transfer.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Flatten()(top_model)
    top_model = Dense(2048, activation="swish")(top_model)
    top_model = Dropout(dropout_rate) (top_model)
    top_model = Dense(1000, activation="softmax")(top_model)
    model = Model(inputs=transfer.input, outputs = top_model)
    return model

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.06)
path = '../data/LPD_competition/cp/cp_0324_1_resnet.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch)
pred_generator = x_pred

model = my_model()
# model.summary()

#3. Compile, Train, Evaluate
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit_generator(train_generator, epochs=200, steps_per_epoch = len(x_train) // batch ,
    validation_data=valid_generator, validation_steps=10 , callbacks=[es, lr, cp])

model.save_weights('../data/LPD_competition/cp/cp_0324_1_resnet_weights.h5')

result = model.evaluate(valid_generator, batch_size=batch)
print("loss ", result[0])
print("acc ", result[1])

# EarlyStopping 82
# loss  0.016353940591216087
# acc  0.9954252243041992

#4. Predict
model = load_model('../data/LPD_competition/cp/cp_0324_1_resnet.hdf5')
# model.load_weights('../data/LPD_competition/cp/cp_0320_1_resnet_kf_weights_4.h5')

print(">>>>>>>>>>>>>>>> predict >>>>>>>>>>>>>> ")
# tta 나만 터져?

result = model.predict(pred_generator, verbose=True)

# save
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))
result_arg = np.argmax(result, axis = 1)

submission['prediction'] = result_arg
submission.to_csv('../data/LPD_competition/sub_0324_1.csv', index=True)
# score 59.639


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >>  2:57:19.115280



