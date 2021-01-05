import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. Data
x_train = np.array([1,2,3])
y_train = np.array([1,2,3])

x_test = np.array([6,7,8])
y_test = np.array([6,7,8])

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile & Train
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit (x_train, y_train, epochs=100, batch_size=1)

#4. Evaluate & predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss(mse), metrics(mae) : ", loss)

result = model.predict(x_train)
print("result : ", result)  