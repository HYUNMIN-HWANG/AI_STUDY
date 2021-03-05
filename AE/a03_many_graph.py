# AutoEncoder (비지도 학습)
# input과 output이 동일함 >> 특징을 추출한 후 다시 원래 사이즈로 바꿔준다. >> 잡음제거하는 효과
# 히든레이어 개수 변화에 따른 이미지 차이 확인

import numpy as np
from tensorflow.keras.datasets import mnist

# DATA
# y는 사용하지 않는다.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255
print(x_train.shape, x_test.shape)

# print(x_train[0])
# print(x_test[0])

# Modeling
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Input, Dense

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))

    return model

model_01 = autoencoder(hidden_layer_size=1)
model_02 = autoencoder(hidden_layer_size=2)
model_04 = autoencoder(hidden_layer_size=4)
model_08 = autoencoder(hidden_layer_size=8)
model_16 = autoencoder(hidden_layer_size=16)
model_32 = autoencoder(hidden_layer_size=32)

print("########## node 1개 시작 ##########")
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train, x_train, epochs=10)

print("########## node 2개 시작 ##########")
model_02.compile(optimizer='adam', loss='binary_crossentropy')
model_02.fit(x_train, x_train, epochs=10)

print("########## node 4개 시작 ##########")
model_04.compile(optimizer='adam', loss='binary_crossentropy')
model_04.fit(x_train, x_train, epochs=10)

print("########## node 8개 시작 ##########")
model_08.compile(optimizer='adam', loss='binary_crossentropy')
model_08.fit(x_train, x_train, epochs=10)

print("########## node16개 시작 ##########")
model_16.compile(optimizer='adam', loss='binary_crossentropy')
model_16.fit(x_train, x_train, epochs=10)

print("########## node 32개 시작 ##########")
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train, x_train, epochs=10)

output_01 = model_01.predict(x_test)
output_02 = model_02.predict(x_test)
output_04 = model_04.predict(x_test)
output_08 = model_08.predict(x_test)
output_16 = model_16.predict(x_test)
output_32 = model_32.predict(x_test)

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize=(15,15))
random_imgs = random.sample(range(output_01.shape[0]),5)
outputs = [x_test, output_01, output_02, output_04, output_08, output_16, output_32]

for row_num, row in enumerate(axes) :
    for col_num, ax in enumerate(row) :
        ax.imshow(outputs[row_num][random_imgs[col_num]].reshape(28,28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.show()

# 맨 위가 원본 이미지
# 약 95% (174) 정도면 거의 원본과 유사한 이미지가 나온다. >> PCA와 유사한 개념
