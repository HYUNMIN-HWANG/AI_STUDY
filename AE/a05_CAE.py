# AutoEncoder (비지도 학습)
# input과 output이 동일함 >> 특징을 추출한 후 다시 원래 사이즈로 바꿔준다. >> 잡음제거하는 효과
# Deep learning 구성 (CNN - shape(28,28,3)이 바뀐다.)

## [1] 원칙 : encoder 부분과 decoder 부분 구성을 동일하게 짜야 한다.
## [2] 변이 : 마음대로 모델을 짠다.

import numpy as np
from tensorflow.keras.datasets import mnist

# DATA
# y는 사용하지 않는다.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_train_2 = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
print(x_train.shape, x_test.shape)

# 60000, 28, 28, 1) (10000, 28, 28, 1)
# print(x_train[0])
# print(x_test[0])

# Modeling
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPool2D, UpSampling2D, Conv2DTranspose

# [1] 원칙 : encoder 부분과 decoder 부분 구성을 동일하게 짜야 한다.
# def autoencoder(hidden_layer_size) :
#     model = Sequential()
#     model.add(Conv2D(filters=hidden_layer_size, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))
#     model.add(Conv2D(128,3,padding='same', activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(units=784, activation='sigmoid'))
#     return model

# def autoencoder(hidden_layer_size) :
#     input_img = Input(shape=(28, 28, 1))

#     x = Conv2D(filters=hidden_layer_size, kernel_size=3, padding='same', activation='relu') (input_img)
#     x = MaxPool2D(2, padding='same')(x)
#     x = Conv2D(128,3,padding='same', activation='relu') (x)
#     x = MaxPool2D(2, padding='same')(x)
#     x = Conv2D(64,3,padding='same', activation='relu') (x)
#     encode = MaxPool2D(2, padding='same')(x)

#     x = Conv2D(64,3,padding='same', activation='relu') (encode)
#     x = UpSampling2D(2)(x)
#     x = Conv2D(128,3,padding='same', activation='relu') (x)
#     x = UpSampling2D(2)(x)
#     x = Conv2D(256,3, activation='relu') (x)
#     x = UpSampling2D(2)(x)
#     decode = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

#     model = Model (input_img, decode)
#     return model

def autoencoder(hidden_layer_size) :
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(filters=hidden_layer_size, kernel_size=3, padding='same', activation='relu') (input_img)
    x = MaxPool2D(2, padding='same')(x)
    x = Conv2D(128,3,padding='same', activation='relu') (x)
    x = MaxPool2D(2, padding='same')(x)
    x = Conv2D(64,3,padding='same', activation='relu') (x)
    encode = MaxPool2D(2, padding='same')(x)

    x = Conv2DTranspose(64,3,padding='same', activation='relu') (encode)
    x = UpSampling2D(2)(x)
    x = Conv2D(128,3,padding='same', activation='relu') (x)
    x = UpSampling2D(2)(x)
    x = Conv2D(256,3, activation='relu') (x)
    x = UpSampling2D(2)(x)
    decode = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    model = Model (input_img, decode)
    return model

#####################################################################################

# [2] 변이 : 마음대로 모델을 짠다.
# def autoencoder(hidden_layer_size) :
#     model = Sequential()
#     model.add(Dense(units=hidden_layer_size, input_shape=(784,),activation='relu'))
#     model.add(Dense(64))
#     model.add(Dense(64))
#     model.add(Dense(64))
#     model.add(Dense(64))
#     model.add(Dense(units=784, activation='sigmoid'))
#     return model
# Epoch 10/10
# 1875/1875 [==============================] - 2s 1ms/step - loss: 0.0833 - acc: 0.0138

# 중간 레이어를 크게 잡을 수록 원본 이미지와 유사함, 작게 잡을수록 손실이 많아진다.
model = autoencoder(hidden_layer_size=256)  
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['acc'])
# model.fit(x_train, x_train_2, epochs=10)
model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

# 이미지 확인
from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2, 5, figsize=(20,7))

# 이미지 다섯 개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# autoencoder가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
