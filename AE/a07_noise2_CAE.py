# autoencoder , CNN
# NOISE 제거


import numpy as np
from tensorflow.keras.datasets import mnist

# DATA
# y는 사용하지 않는다.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_train_CNN = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 784).astype('float32')/255
x_test_CNN = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)

# 노이즈 만들기 : 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape) # 0부터 0.1 사이 밝기를 가진 점을 이미지에 랜덤하게 찍어준다. 
x_train_noised_CNN = x_train_CNN + np.random.normal(0, 0.1, size=x_train_CNN.shape) # 0부터 0.1 사이 밝기를 가진 점을 이미지에 랜덤하게 찍어준다. 
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_test_noised_CNN = x_test_CNN + np.random.normal(0, 0.1, size=x_test_CNN.shape)
# 문제점 >> MinMax가 안 맞게 된다.
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)      
x_train_noised_CNN = np.clip(x_train_noised_CNN, a_min=0, a_max=1)     
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
x_test_noised_CNN = np.clip(x_test_noised_CNN, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Flatten

def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(154, 3, padding='same', activation='relu'))
    model.add(Conv2D(1, 3, padding='same', activation='sigmoid'))
    return model

# def autoencoder(hidden_layer_size):
#     model = Sequential()
#     model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding = 'same', input_shape=(28,28,1)))
#     model.add(MaxPool2D(2))
#     # model.add(BatchNormalization())
#     model.add(Conv2D(16, 2, 1))
#     model.add(UpSampling2D((2, 2))) # 맥스풀 했으니 업 해줘야 한다.
#     model.add(Flatten())
#     model.add(Dense(units=hidden_layer_size/2))
#     model.add(Dense(units=hidden_layer_size))
#     model.add(Dense(units=784, activation='sigmoid'))
#     return model

model = autoencoder(hidden_layer_size=154)  # 154 : 95% PCA 에 해당하는 값 : 가장 원본과 비슷하게 복원이 되는지 확인하기 위함
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(x_train_noised_CNN, x_train, epochs=50, batch_size=16)  # x : noise 이미지, y : 원본 이미지
model.fit(x_train_noised_CNN, x_train_CNN, epochs=5, batch_size=4)  # x : noise 이미지, y : 원본 이미지

output = model.predict(x_test_noised_CNN)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이지 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# autoencoder 한 후 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(output[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()