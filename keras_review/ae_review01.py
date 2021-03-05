import numpy as np
from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, _), (x_test, _ ) = mnist.load_data()
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)

# preprocessing
x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense

def autoencoder(size) :
    input_img = Input(shape=(784,))

    x = Dense(size, activation='relu') (input_img)
    decoder = Dense(784, activation='sigmoid') (x)

    model = Model(input_img, decoder)
    return model

model = autoencoder(128)
# model.summary()

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train, x_train, batch_size=16, epochs=5)

#4. Evaluate, Predict
result = model.evaluate(x_test, x_test, batch_size=16)
print(result)

output = model.predict(x_test)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),(ax6, ax7, ax8, ax9, ax10)) =\
    plt.subplots(2, 5, figsize=(20,7))

random_images = random.sample(range(output.shape[0]),5)

# 원본
for i , ax in enumerate([ax1, ax2, ax3, ax4, ax5]) :
    print(i, ax)
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("INPUT", size=15)
    ax.grid(False)
    ax.set_xticks([])   
    ax.set_yticks([])   

# output
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]) :
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i == 0 :
        ax.set_ylabel("OUTPUT", size=15)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()