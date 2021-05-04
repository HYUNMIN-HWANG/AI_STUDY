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
    # overwrite=True,
    max_trials=2    # epochs만큼을 2번 반복했다.
)
# 특징 1 : input shape 지정을 안 했는데도 돌아간다.
# 특징 2 : y 데이터에 onehotencoding 안했는데도 돌아감
# 특징 3 : y 데이터에 onehotencoding 한 후도 잘 돌아감 < onehotencoding을 해도 되고 안해도 된다.
# 특징 4 : 체크포인트가 자동으로 생성된다.

model.fit(x_train, y_train, epochs=3)

results = model.evaluate(x_test, y_test)

print(results)  # [0.042958956211805344, 0.9855999946594238]

