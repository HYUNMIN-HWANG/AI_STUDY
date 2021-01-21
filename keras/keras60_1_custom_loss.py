# loss 함수를 직접 만들어본다.


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

def custom_mse(y_true, y_pred) :    # y_true : y의 원래 값 // y_pred :fit 해서 나온 y 값
    return tf.math.reduce_mean(tf.square(y_true - y_pred))   # mse

#1. DATA
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')

# print(x.shape)  # (8,)

#2. Modeling
model = Sequential()
model.add(Dense(10, input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))


#3. Compile, Train
model.compile(loss= custom_mse, optimizer='adam')   
model.fit(x, y, batch_size=1, epochs=30)

#4. Evaluate, Predict

loss = model.evaluate(x, y)
print(loss)
# 0.37671029567718506