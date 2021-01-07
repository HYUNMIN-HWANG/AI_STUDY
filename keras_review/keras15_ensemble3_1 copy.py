# ensemble
# model1, model2 -> concatenate -> model1, model2, model3

import numpy as np

#1. DATA
x1 = np.array ([range(100), range(101, 201), range(201,301), range(301, 401), range(401, 501)]) #(5, 100)
x2 = np.array ([range(20,120), range(40, 140), range(60, 160), range(80, 180), range(100,200)]) #(5, 100)

y1 = np.array ([range(10,110), range(111, 211), range(211,311), range(311, 411), range(411, 511)]) #(5, 100)
y2 = np.array ([range(20,120), range(121, 221), range(221,321), range(321, 421), range(421, 521)]) #(5, 100)
y3 = np.array ([range(100), range(101, 201), range(201,301), range(301, 401), range(401, 501)]) #(5, 100)

x1 = np.transpose(x1)
x2 = np.transpose(x2)

y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test = \
    train_test_split(x1, x2, y1, train_size=0.8, shuffle=False)
y2_train, y2_test, y3_train, y3_test = train_test_split(y2, y3, train_size=0.8, shuffle=False)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# Model 1
input1 = Input(shape=(5,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10,activation='relu')(dense1)
dense1 = Dense(10,activation='relu')(dense1)

# Mode1 2
input2 = Input(shape=(5,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(10,activation='relu')(dense2)
dense2 = Dense(10,activation='relu')(dense2)

# model merge
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(middle1)

# model dense
output1 = Dense(20)(middle1)
output1 = Dense(20)(output1)
output1 = Dense(5)(output1)

output2 = Dense(30)(middle1)
output2 = Dense(30)(output2)
output2 = Dense(5)(output2)

output3 = Dense(40)(middle1)
output3 = Dense(40)(output3)
output3 = Dense(5)(output3)

# 모델 선언
model = Model(inputs = [input1, input2], outputs = [output1,output2,output3])
model.summary()

#3. Compile, Train
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train,y2_train, y3_train],\
         epochs=50, batch_size=1, validation_split=0.2, verbose=0)

#4. Evaluate, Predict
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test, y3_test], batch_size=1)
print("loss : ", loss)

y_predict1, y_predict2, y_predict3 = model.predict([x1_test, x2_test])
print("y_predict1 : \n", y_predict1)      
print("y_predict1 : \n", y_predict2)      
print("y_predict1 : \n", y_predict3)      

# RMSE

from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))

RMSE1 = RMSE(y1_test, y_predict1)
RMSE2 = RMSE(y2_test, y_predict2)
RMSE3 = RMSE(y3_test, y_predict3)
RMSE = (RMSE1 + RMSE2 + RMSE3)/3 #전체 RMSE
print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE3 : ", RMSE3)
print("RMSE : ", RMSE)

# R2
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y_predict1)
r2_2 = r2_score(y2_test, y_predict2)
r2_3 = r2_score(y3_test, y_predict3)
r2 = (r2_1 + r2_2 + r2_3) / 3  #전체 r2
print("R2_1 : ", r2_1)
print("R2_2 : ", r2_2)
print("R2_3 : ", r2_3)
print("R2 : ", r2)