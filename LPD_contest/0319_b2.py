import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB0, InceptionV3, MobileNet, ResNet50, ResNet101, EfficientNetB2
# from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
# from tensorflow.keras.applications.resnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


start_now = datetime.datetime.now()

#1. DATA
### npy load
x_data = np.load('../data/LPD_competition/npy/data_x_train4.npy', allow_pickle=True)
print(x_data.shape)    # (48000, 100, 100, 3)
y_data = np.load('../data/LPD_competition/npy/data_y_train4.npy', allow_pickle=True)
print(y_data.shape)    # (48000,)
x_pred = np.load('../data/LPD_competition/npy/data_x_pred4.npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 100, 100, 3)

# preprocess
x_data = preprocess_input(x_data)
x_pred = preprocess_input(x_pred)

y_data = to_categorical(y_data)

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=42)

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=20,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

batch = 32
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch)
valid_generator = test_datagen.flow(x_valid, y_valid, batch_size=batch)
pred_generator = test_datagen.flow(x_pred, shuffle=False)

#2. Modeling
def model() :
    ef = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(100, 100, 3))
    top_model = ef.output
    top_model = Flatten()(top_model)
    # top_model = Dense(1024, activation="relu")(top_model)
    # top_model = Dropout(0.2)(top_model)
    top_model = Dense(1000, activation="softmax")(top_model)

    model = Model(inputs=ef.input, outputs = top_model)
    model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.9), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model 


#3. Compile, Train, Evaluate
es = EarlyStopping(monitor='val_loss', patience=15, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=8, factor=0.06)
path = '../data/LPD_competition/cp/cp_0319_1_b2.hdf5'
cp = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')

model = model()
model.fit_generator(train_generator, epochs=100, steps_per_epoch = len(x_train) // batch ,
    validation_data=valid_generator, callbacks=[es, lr, cp])

model.save_weights('../data/LPD_competition/cp/cp_0319_1_b2_weights.h5')

result = model.evaluate(valid_generator, batch_size=batch)
print("loss ", result[0])
print("acc ", result[1])


#4. Predict

print("predict >>>>>>>>>>>>>> ")
submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

model = load_model('../data/LPD_competition/cp/cp_0319_1_b2.hdf5')

result = model.predict(pred_generator, verbose=True)
print(result.shape) # (72000, 1000)
print(np.argmax(result, axis = 1))

submission['prediction'] = np.argmax(result, axis = 1)
submission.to_csv('../data/LPD_competition/sub_0319_1.csv',index=True)


end_now = datetime.datetime.now()
time = end_now - start_now
print("time >> " , time)    # time >> 

# score 
