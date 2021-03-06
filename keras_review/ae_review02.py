# autoencoder
# NOISE 제거

import numpy as np
from tensorflow.keras.datasets import mnist

# DATA
# y는 사용하지 않는다.
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, _), (x_test, _) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

x_train_d = x_train.reshape(60000, 784).astype('float32')/255
x_test_d = x_test.reshape(10000, 784).astype('float32')/255
x_train_c = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test_c = x_test.reshape(10000, 28, 28, 1).astype('float32')/255
print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

x_train_noised = x_train_d + np.random.normal(0, 0.1, size=x_train_d.shape)
x_test_noised = x_test_d + np.random.normal(0, 0.1, size=x_test_d.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

x_train_noised_c = x_train_c + np.random.normal(0, 0.1, size=x_train_c.shape)
x_test_noised_c = x_test_c + np.random.normal(0, 0.1, size=x_test_c.shape)
x_train_noised_c = np.clip(x_train_noised_c, a_min=0, a_max=1)
x_test_noised_c = np.clip(x_test_noised_c, a_min=0, a_max=1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def autoencoder (size) :
    # input_img = Input(shape=(784,))
    # x = Dense(size, activation='relu') (input_img)

    input_img = Input(shape=(28,28,1))

    x = Conv2D(size, 3, activation='relu', padding='same') (input_img)
    x = Conv2D(160, 3, activation='relu', padding='same') (x)
    encoder = MaxPool2D(3, padding='same')(x)

    x = Conv2D(160, 3, activation='relu',padding='same' ) (encoder)
    x = UpSampling2D(3) (x)
    x = Conv2D(184, 3, activation='relu') (x)
    output = Conv2D(1, 3, padding='same', activation='sigmoid') (x)

    model = Model(input_img, output)
    return model

model = autoencoder(184)
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train_noised_c, x_train_c, epochs=10)

result = model.evaluate(x_test_noised_c, x_test_c)
print(result)

output = model.predict(x_test_noised)
print(output)

# image
import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),(ax11, ax12, ax13, ax14, ax15)) = \
    plt.subplots(3, 5, figsize=(20,7))
    
random_images = random.sample(range(output.shape[0]),5)

for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 1 :
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


plt.tight_layout()
plt.show()