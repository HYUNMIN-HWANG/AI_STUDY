import numpy as np
import tensorflow as tf

#1. Data
x = np.array(range(1,11))
y = np.array(range(1,11))

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear'))
model.add(Dense(5, activation = 'linear'))
model.add(Dense(1))
model.add(Dense(1))

#3. Compile & Train
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. Evalute
result = model.evaluate(x, y, batch_size=1)
print("result : ", result)

x_pred = np.array([4])
predict = model.predict(x_pred)
print("prediction : ", predict)