# LSTM

#1. DATA
import numpy as np

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])

# print(x.shape)  # (4, 3)
# print(y.shape)  # (4, )

# 2 dim ->3 dim
x = x.reshape(4, 3, 1)

x_pred = np.array([5, 6, 7])    # (3, )
x_pred = x_pred.reshape(1, 3, 1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(40, activation='relu',input_shape=(3, 1)))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print("result : ", y_pred)

# loss :  0.0013327046763151884
# mae :  0.03114783763885498
# result :  [[8.01416]]