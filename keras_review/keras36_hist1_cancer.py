# hist를 이용하여 그래프를 그리시오
# loss, val_loss, acc, val_acc

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

print(x.shape)  # (569, 30)
print(y.shape)  # (569, )

# print(y) 0 or 1 >> binary

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, \
    shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)# (512, 30, 1)
print(x_test.shape) # (57, 30, 1)

#2. Modeling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu',input_shape=(30,1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

#3. Compile. Train
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=2,mode='min')

hist = model.fit(x_test, y_test, epochs=100, batch_size=1, validation_split=0.2, verbose=1,callbacks=[es])

print(hist.history.keys())

# Graph
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('spochs')
plt.legend(['loss','val loss','acc','val_acc'])
plt.show()

#4. Evaluate, Predict
y_pred = model.predict(x_test[-5:-1])
print(y_test[-5:-1])
print(y_pred)

print(np.where(y_pred>0.5,1,0))