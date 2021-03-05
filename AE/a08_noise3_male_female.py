# keras67-1 noise 추가한 후, 다시 원상 복구하는 작업
# numpy 파일에 잡음 넣어서 

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization, AveragePooling2D, Activation, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

#1. DATA
# npy load
x_train = np.load('../data/image/gender/npy/keras67_train_x.npy')
x_valid = np.load('../data/image/gender/npy/keras67_valid_x.npy')

print(x_train.shape, x_valid.shape)  # (1389, 56, 56, 3) (347, 56, 56, 3) 

x_train_output = x_train.reshape(1389, 9408)

# Noise 만들기
x_train_noise_CNN = x_train + np.random.normal(0, 0.2, size=x_train.shape)
x_valid_noise_CNN = x_valid + np.random.normal(0, 0.2, size=x_valid.shape)
# Minmax 맞추기
x_train_noise_CNN = np.clip(x_train_noise_CNN, a_min=0, a_max=1)
x_valid_noise_CNN = np.clip(x_valid_noise_CNN, a_min=0, a_max=1)
print(x_train_noise_CNN.shape, x_valid_noise_CNN.shape)  # (1389, 56, 56, 3) (347, 56, 56, 3)


#2. Modeling
def autoencoder(hidden_layer_size) :
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size, kernel_size=3, padding='same', activation='relu', input_shape=(56, 56, 3)))
    model.add(Conv2D(200, 3, padding='same', activation='relu'))
    model.add(Conv2D(200, 3, padding='same', activation='relu'))
    model.add(Conv2D(281, 3, padding='same', activation='relu'))
    model.add(Conv2D(3, 3, padding='same', activation='sigmoid'))
    # model.add(Flatten())
    # model.add(Dense(200, activation='relu'))
    # model.add(Dense(281, activation='relu'))
    # model.add(Dense(units=9408, activation='sigmoid'))
    return model

model = autoencoder(281)
model.summary()

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(x_train_noise_CNN, x_train_output, epochs=10, batch_size=4, callbacks=[es])  # x : noise 이미지, y : 원본 이미지
model.fit(x_train_noise_CNN, x_train, epochs=10, batch_size=4, callbacks=[es], validation_split=0.2)  # x : noise 이미지, y : 원본 이미지

output = model.predict(x_valid_noise_CNN)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_valid[random_images[i]].reshape(56,56,3), cmap='gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이지 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(x_valid_noise_CNN[random_images[i]].reshape(56,56,3), cmap='gray')
    if i == 0 :
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# autoencoder 한 후 이미지
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]) :
    ax.imshow(output[random_images[i]].reshape(56,56,3), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()