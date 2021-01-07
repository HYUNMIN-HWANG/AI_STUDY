# ensemble
# model1, model2 -> concatenate -> model1, moedl2

import numpy as np

#1. DATA
x1 = np.array ([range(100), range(101, 201), range(201,301), range(301, 401), range(401, 501)])     #(5, 100)
y1 = np.array ([range(10,110), range(111, 211), range(211,311), range(311, 411), range(411, 511)])  #(5, 100)

x2 = np.array ([range(20,120), range(40, 140), range(60, 160), range(80, 180), range(100,200)]) #(5, 100)
y2 = np.array ([range(25,125), range(45, 145), range(65, 165), range(85, 185), range(105,205)]) #(5, 100)

x1 = np.transpose(x1) #(100, 5)
y1 = np.transpose(y1) #(100, 5)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=False)

# train (80, 5)
# test  (20, 5)

x1_predict_made = np.array([501, 601, 701, 801, 901])  #(5, )
x1_predict_made = x1_predict_made.reshape(1,5)
print(x1_predict_made.shape) # (1,5)


#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

# [모델 구성]
# Model1
input1 = Input(shape = (5,)) # input1 = 5
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)

# Moedl2
input2 = Input(shape=(5,)) # input2 = 5
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)

# [모델 병합]
from tensorflow.keras.layers import concatenate
merge1 = concatenate([dense1, dense2])
middle1 = Dense(30)(merge1)
middle1 = Dense(30)(middle1)
middle1 = Dense(30)(middle1)
middle1 = Dense(30)(middle1)

# [모델 분기]
# Model 1
output1 = Dense(20)(middle1)
output1 = Dense(20)(output1)
output1 = Dense(20)(output1)
output1 = Dense(5)(output1) # 최종 output = 5

# Model 2
output2 = Dense(30)(middle1)
output2 = Dense(30)(output2)
output2 = Dense(30)(output2)
output2 = Dense(5)(output2) # 최종 output = 5

# [모델 선언]
model = Model(inputs = [input1, input2], outputs = [output1,output2])
model.summary()

#3. Conpile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'])
print("please wait....")
model.fit([x1_train, x2_train],[y1_train,y2_train],epochs=20,batch_size=1,\
            validation_split=0.2, verbose=1)

#4. Evaluate, Predict
loss = model.evaluate([x1_test, x2_test],[y1_test,y2_test],batch_size=1)
print("loss : ", loss)

y1_predict, y2_predict = model.predict([x1_test,x2_test])
# print("====================================================")
# print("y1_predict : \n", y1_predict)
# print("y2_predict : \n", y2_predict)
# print(y2_predict.shape) #(20, 5)
# print("====================================================")

# 인풋, 아웃풋이 2개 이므로 predict도 2개를 넣어야 한다.
y3_predict_made,y3_predict_made = model.predict([x1_predict_made,x1_predict_made])
print(y3_predict_made)


# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_train) :
    return np.sqrt(mean_squared_error(y_test,y_train))

RMSE1 = RMSE(y1_test, y1_predict)
RMSE2 = RMSE(y2_test, y2_predict)
RMSE = (RMSE1+RMSE2)/2

print("RMSE1 : ", RMSE1)
print("RMSE2 : ", RMSE2)
print("RMSE : ", RMSE)

# R2
from sklearn.metrics import r2_score
r2_1 = r2_score(y1_test, y1_predict)
r2_2 = r2_score(y2_test, y2_predict)
r2 = (r2_1 + r2_2)/2

print("r2_1 : ", r2_1)
print("r2_2 : ", r2_2)
print("r2 : ", r2)
