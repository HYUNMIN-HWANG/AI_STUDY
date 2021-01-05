#1. DATA
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

x = x.reshape(4, 3, 1)

x_pred = np.array([5,6,7])
x_pred = x_pred.reshape(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(40, activation='relu', input_length = 3, input_dim = 1))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1))

#3. Compile, Trian
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x, y, batch_size=1)
print(loss, mae)

y_pred = model.predict(x_pred)
print(y_pred)

# 0.004362011328339577 0.061776041984558105
# [[8.203011]]