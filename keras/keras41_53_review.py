# CNN

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)  # (150, 4)
print(y.shape)  # (150, ) 
# print(y)        # 다중분류 >> 0, 1, 2

df = pd.DataFrame(x, columns=dataset['feature_names'])
print(df)

print(df.shape)

print(df.columns)

print(df.index)

print(df.head())

print(df.tail())

print(df.info())

print(df.describe())

print(df.isnull())


# x > processing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=55)
x_train, x_val, y_train, y_val = train_test_split (x_train, y_train, train_size=0.8, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], 2, 2, 1)
x_test = x_test.reshape(x_test.shape[0], 2, 2, 1)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# print(y_train.shape)    # (120, 3)
# print(y_test.shape)     # (30, 3)

#2. Modeling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Flatten

model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), strides=2, padding='same',\
    input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3])))
model.add(Conv2D(filters=65, kernel_size=(2,2), padding='same'))
# model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))

model.summary()

# 1
model.save('../data/h5/model1.h5')
model = load_model('../data/h5/model1.h5')


'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 1, 1, 128)         640
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 1, 1, 65)          33345
_________________________________________________________________
flatten (Flatten)            (None, 65)                0
_________________________________________________________________
dense (Dense)                (None, 64)                4224
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 99
=================================================================
Total params: 40,388
Trainable params: 40,388
Non-trainable params: 0
_________________________________________________________________
'''

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

save = '../modelcheckpoint/{epoch:2d}-{val_loss:.4f}.hd5f'
es = EarlyStopping(monitor='val_loss', patience=5, mode='min')
cp = ModelCheckpoint(filepath=save, monitor='val_loss', save_best_only=True, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=5, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=16, \
    validation_data=(x_val, y_val), verbose=1, callbacks=[es, lr, cp])

# 2
model.save('../data/h5/model.h5')
model = load_model('../data/h5/model.h5')

# 가중치 저장
model.save_weights('../data/h5/k52.h5')
model.load_weights('../data/h5/k52.h5')

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print("loss : ", loss)
print("acc : ", acc)


print("y_test : ", np.argmax(y_test[-5:], axis=1))

y_pred = model.predict(x_test)
print("y_pred : ", np.argmax(y_pred[-5:], axis=1))


