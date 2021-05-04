# 과제 : 회귀
# trail은 2 이상

import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# DATA
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, shuffle=True)

print(x_train.shape, x_test. shape) # (404, 13) (102, 13)
print(y_train.shape, y_test.shape)  # (404,) (102,)  

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

# Model
# model = ak.ImageRegressor(
#     overwrite=True,
#     max_trials=1,
#     metrics=['mse']
# )
# shape 안 맞음
# ValueError: Expect the data to ImageInput to have shape (batch_size, height, width, channels) or (batch_size, height, width) dimensions, but got input shape [32, 13]

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=2,
    metrics=['mse']
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=6, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, mode='min')
path = '../data/cp/'
cp = ModelCheckpoint(filepath = path, monitor='val_loss', save_weights_only=True, save_best_only=True)

model.fit(x_train, y_train, epochs=10, validation_split=0.2, 
          callbacks=[es, lr, cp])

results = model.evaluate(x_test, y_test)
print(results)  # [126.91399383544922, 126.91399383544922]

best_model = model.export_model()
try :
    best_model.save('../data/autokeras/best_model_boston', save_format='tf')
except :
    best_model.save('../data/autokeras/best_model_boston.h5')

# from tensorflow.keras.models import load_model

# model = load_model('../data/autokeras/best_model_boston', custom_objects=ak.CUSTOM_OBJECTS)
# model.summary()

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 13)]              0
_________________________________________________________________
multi_category_encoding (Mul (None, 13)                0
_________________________________________________________________
normalization (Normalization (None, 13)                27
_________________________________________________________________
dense (Dense)                (None, 32)                448
_________________________________________________________________
re_lu (ReLU)                 (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056
_________________________________________________________________
re_lu_1 (ReLU)               (None, 32)                0
_________________________________________________________________
regression_head_1 (Dense)    (None, 1)                 33
=================================================================
Total params: 1,564
Trainable params: 1,537
Non-trainable params: 27
_________________________________________________________________
'''