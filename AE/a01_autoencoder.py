# AutoEncoder (비지도 학습)
# input과 output이 동일함 >> 특징을 추출한 후 다시 원래 사이즈로 바꿔준다. >> 잡음제거하는 효과

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

input_img = Input(shape=(784,))                     # input
encoded = Dense(64, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded) # output ***activation='sigmoid'***
# decoded = Dense(784, activation='relu')(encoded) # relu 문제점 : 음수에 해당하는 값들은 계산하지 않아서 학습이 덜 된다.

autoencoder = Model(input_img, decoded)

autoencoder.summary()

# Compile, Train
# autoencoder는 acc가 아닌 loss로 지표를 확인한다. 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # acc 잘 안나온다. : 784가 output인데 sigmoid를 사용했기 때문
# autoencoder.compile(optimizer='adam', loss='mse')   # 이것도 가능하다 : 어차피 이미지는 0과 1사이로 나오기 때문
# autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])   # acc 잘 안나온다. : 784가 output인데 sigmoid를 사용했기 때문

autoencoder.fit(x_train, x_train, epochs=30, batch_size=256, validation_split=0.2)  # x와 y가 동일함

decoded_img = autoencoder.predict(x_test)

# 이미지 확인
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4))
for i in range(n) :
    ax = plt.subplot(2, n, i+1)         # 원래 이미지
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))  # 디코드한 이미지
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()