# CNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

model = Sequential()
model.add(Conv2D(10, (2,2), strides=2,padding='same',input_shape=(10, 10, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(9, (2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(1))

model.summary()
