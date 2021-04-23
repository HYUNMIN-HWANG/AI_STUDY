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

# submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
# print(submission.shape) # (72000, 2)

start_now = datetime.datetime.now()

### npy load
x_data = np.load('../data/LPD_competition/npy/data_x_train4.npy', allow_pickle=True)
print(x_data.shape)    # (48000, 100, 100, 3)
y_data = np.load('../data/LPD_competition/npy/data_y_train4.npy', allow_pickle=True)
print(y_data.shape)    # (48000,)
x_pred = np.load('../data/LPD_competition/npy/data_x_pred4.npy', allow_pickle=True)
print(x_pred.shape)     # (72000, 100, 100, 3)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True, random_state=42)

x_data = preprocess_input(x_data)
x_pred = preprocess_input(x_pred)

y_data = to_categorical(y_data)

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_rate=20,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

def model(node=512, drop=0.2) :
    ef = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(100, 100, 3))

    top_model = ef.output
    top_model = Flatten()(top_model)
    top_model = Dense(node, activation="relu")(top_model)
    top_model = Dropout(drop)(top_model)
    top_model = Dense(1000, activation="softmax")(top_model)

    model = Model(inputs=ef.input, outputs = top_model)
    model.compile(optimizer=SGD(learning_rate=0.005, momentum=0.9), loss = 'categorical_crossentropy', metrics=['accuracy'])
    return model 

model2 = KerasClassifier(build_fn= model , verbose=1)  

def create_hyperparameters() :
    node = [1024, 512, 128, 64, 32]
    # learning_rate = [0.1, 0.5, 0.05, 0.01, 0.005, 0.001]
    drop = [0.2, 0.3, 0.4, 0.5]
    # return {"node" : node, "lr" : learning_rate, "drop" : drop}
    # return {"lr" : learning_rate}
    return {"node" : node, 'drop' : drop}

hyperparameters = create_hyperparameters()

# search = RandomizedSearchCV(model2, hyperparameters, cv=4)
search = GridSearchCV(model2, hyperparameters, cv=4)

search.fit(x_train, y_train, verbose=1)

print("best_params : ", search.best_params_)  
# best_params :  {'lr': 0.005}

acc = search.score(x_test, y_test)
print("Score : ", acc)
# Score :  0.9728124737739563 