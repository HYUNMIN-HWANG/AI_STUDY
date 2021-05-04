# autokeras 
# pip install autokeras
# model = ak.ImageClassifier

import numpy as np
import tensorflow as tf
import autokeras as ak

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255.

print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28, 1) (10000,)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,    # epochs만큼을 2번 반복했다.
    loss='mse',
    # metrcis=['mse']   # 회귀모델일 때
    metrics=['acc']
)

# model.summary()   

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', patience=6, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, mode='min')
path = '../data/cp/'
cp = ModelCheckpoint(filepath = path, monitor='val_loss', save_weights_only=True, save_best_only=True)

model.fit(x_train, y_train, epochs=1, validation_split=0.2, 
          callbacks=[es, lr, cp])

results = model.evaluate(x_test, y_test)

print(results)  # [0.05430540069937706, 0.9819999933242798] <<- [loss, acc]

# model.summary()

# 모델 저장하는 두 가지 방법
# [1] 모델 저장
model2 = model.export_model()
model2.save('../data/autokeras/save_model.h5')   

# [2] 가장 좋은 모델을 저장
best_model = model.tuner.get_best_model()
best_model.save('../data/autokeras/best_model.h5')

