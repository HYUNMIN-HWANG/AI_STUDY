# 과제 : 이진 분류
# trail은 2 이상


import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# DATA
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)

print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape) # (114, 30) (114,)

scale = MinMaxScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)


# autokeras model : data shape에 맞는 모델을 넣어야 한다.
# model = ak.ImageClassifier(
#     overwrite=True,
#     max_trials=1,    
#     loss='mse',
#     metrics=['acc']
# )
# shape이 맞지 않으면 아래와 같은 에러가 난다.
# ValueError: Expect the data to ImageInput to have shape (batch_size, height, width, channels) or (batch_size, height, width) dimensions, but got input shape [32, 30]

model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=2
)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=6, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, mode='min')
path = '../data/cp/'
cp = ModelCheckpoint(filepath = path, monitor='val_loss', save_weights_only=True, save_best_only=True)

model.fit(x_train, y_train, epochs=10, validation_split=0.2, 
          callbacks=[es, lr, cp])

results = model.evaluate(x_test, y_test)

print(results)      # [0.06463649868965149, 0.9824561476707458]

best_model = model.tuner.get_best_model()
best_model.save('../data/autokeras/best_model_cancer.h5')

# from tensorflow.keras.models import load_model
# model = load_model('../data/autokeras/best_model_cancer.h5')
# model.summary()

'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 30)]              0
_________________________________________________________________
multi_category_encoding (Mul (None, 30)                0
_________________________________________________________________
normalization (Normalization (None, 30)                61
_________________________________________________________________
dense (Dense)                (None, 32)                992
_________________________________________________________________
re_lu (ReLU)                 (None, 32)                0
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056
_________________________________________________________________
re_lu_1 (ReLU)               (None, 32)                0
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
_________________________________________________________________
classification_head_1 (Activ (None, 1)                 0
=================================================================
Total params: 2,142
Trainable params: 2,081
Non-trainable params: 61
_________________________________________________________________
'''