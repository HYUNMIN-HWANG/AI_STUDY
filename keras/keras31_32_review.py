# LSTM

import numpy as np

def split_x (seq, size) :
    aaa = []
    for i in range(len(seq) - size + 1) :
        subset = seq[i : (i+size)]
        aaa.append(subset)
    print(type(aaa))
    return np.array(aaa)

a = np.array(range(1,11))
size=5

dataset = split_x(a, size)

# print(dataset)
'''
[[ 1  2  3  4  5]
 [ 2  3  4  5  6]
 [ 3  4  5  6  7]
 [ 4  5  6  7  8]
 [ 5  6  7  8  9]
 [ 6  7  8  9 10]
'''

#1. DATA
x = dataset[:,:4]
# print(x)
# print(x.shape)  # (6, 4)
x = x.reshape(x.shape[0],x.shape[1],1)  # (6, 4, 1)
y = dataset[:,-1]
# print(y)  # [ 5  6  7  8  9 10]
# print(y.shape)  # (6,)
x_pred = dataset[-1:,1:]    #(1, 4)
x_pred = x_pred.reshape(x_pred.shape[0],x_pred.shape[1],1)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(20, activation='relu', input_shape=(4, 1)))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(1))

# model.summary()

#3. Compile, Train
model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

#4. Evaluate, Predict
loss, mae = model.evaluate(x, y, batch_size=1)
print("loss : ", loss)
print("mae : ", mae)

y_pred = model.predict(x_pred)
print("y_pred : ", y_pred)