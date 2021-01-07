from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from numpy import array

#1. DATA
x = np.array(range(1,101))
y = np.array(range(1,101))

from sklearn.model_selection import train_test_split
#전체 데이터 == train 80 / test20
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x = np.array(range(1,101))
# train 데이터 == train 80 / validation 20
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)
print(x_train.shape) #(64,)
print(x_test.shape)  #(20,)
print(x_val.shape)   #(16,)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)
print(y_predict)