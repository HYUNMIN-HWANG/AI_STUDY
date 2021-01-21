# quantile loss 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K

# def custom_mse(y_true, y_pred) :    # y_true : y의 원래 값 // y_pred :fit 해서 나온 y 값
#     return tf.math.reduce_mean(tf.square(y_true - y_pred))   # mse

#  Pinball loss
def quantile_loss(y_true, y_pred) :
    qs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # list
    q = tf.constant(np.array([qs]), dtype=tf.float32) # constant : list -> tensorflow의 상수(바뀌지 않는 값)로 바꾼다.
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)   
    return K.mean(v)    # 평균 하나만 출력된다. # 0.5 : 중간값, mae과 유사하다.

# Quantile loss definition
def quantile_loss_dacon(q, y_true, y_pred):
	err = (y_true - y_pred)
	return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
# model.compile(loss=quantile_loss , optimizer='adam')   
for i in range(9) :
    print(quantiles[i])
    model.compile(loss = lambda y_true, y_pred: quantile_loss_dacon(quantiles[i], y_true, y_pred), optimizer='adam')
    # lamda == df
    # y_true, y_pred : input
    model.fit(x, y, batch_size=1, epochs=30)

    #4. Evaluate, Predict
    loss = model.evaluate(x, y)
    print(loss)

# mse loss :  0.37671029567718506
# quantile loss : 0.03600059449672699
# quantile[0] : 0.016497798264026642