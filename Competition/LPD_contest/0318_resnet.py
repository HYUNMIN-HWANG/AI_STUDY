import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 

start_now = datetime.datetime.now()

### npy load
x_train = np.load('../data/LPD_competition/npy/data_x_train4.npy', allow_pickle=True)
print(x_train.shape)    # (48000, 100, 100, 3)
y_train = np.load('../data/LPD_competition/npy/data_y_train4.npy', allow_pickle=True)
print(y_train.shape)    # (48000,)
x_pred = np.load('../data/LPD_competition/npy/data_x_pred4.npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 100, 100, 3)


#1. DATA
# preprocess
x_train = preprocess_input(x_train)
x_pred = preprocess_input(x_pred)

y_train = to_categorical(y_train)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=42)
print(x_train.shape, x_valid.shape)  # (38400, 100, 100, 3) (9600, 100, 100, 3)
print(y_train.shape, y_valid.shape)  # (38400, 1000) (9600, 1000)


train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

batch = 16
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch, shuffle=False )
valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch, shuffle=False )
pred_generator = test_datagen.flow(x_pred, shuffle=False)

'''
#2. Modeling
transfer = ResNet50(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
for layer in transfer.layers:
        layer.trainable = False
top_model = transfer.output
top_model = GlobalAveragePooling2D()(top_model)
top_model = Flatten()(top_model)
top_model = Dense(1024, activation="relu")(top_model)
top_model = Dropout(0.2) (top_model)
top_model = Dense(1000, activation="softmax")(top_model)
model = Model(inputs=transfer.input, outputs = top_model)
model.summary()

#3. Compile, Train, Evaluate
es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.06)
path = '../data/LPD_competition/cp/cp_0318_4_resnet_{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
model.fit_generator(train_generator, epochs=10, steps_per_epoch = len(x_train) // batch ,
    validation_data=valid_generator, callbacks=[es, lr, cp])

model.save('../data/LPD_competition/cp/cp_0318_4_resnet_model.hdf5')
model.save_weights('../data/LPD_competition/cp/cp_0318_4_resnet_weights.h5')

result = model.evaluate(valid_generator, batch_size=batch)
print("loss ", result[0])
print("acc ", result[1])

'''
#4. Predict
submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

model = load_model('../data/LPD_competition/cp/cp_0318_4_resnet_0.6817.hdf5')
model.summary()

result = model.evaluate(valid_generator, batch_size=batch)
print("loss ", result[0])
print("acc ", result[1])

# loss  0.6817045211791992
# acc  0.8711458444595337

print("predict >>>>>>>>>>>>>> ")

result = model.predict(pred_generator, verbose=True)
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))

submission['prediction'] = np.argmax(result, axis = 1)
submission.to_csv('../data/LPD_competition/sub_0318_4.csv',index=True)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >

# train, valid, pred > shuffle=False
# score 49.478
