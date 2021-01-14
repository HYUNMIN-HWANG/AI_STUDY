# 다음 날 삼성의 '종가'를 예측한다.

import numpy as np

# 전체 DATASET
data = np.load('./samsung/samsung_slicing_data3.npy')
# print(data.shape)   # (2397, 6)

# size : 며칠씩 자를 것인지
# col : 열의 개수

def split_x(seq, col,size) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

size = 6
col = 6
dataset = split_x(data,col,size)
# print(dataset)
# print(dataset.shape) # (2392, 6, 6)

# ================================================

#1. DATA
x = dataset[:-1,:,:7]
# print(x)
# print(x.shape)  # (2391, 6, 6)

y = dataset[1:,-1:,-1:]
# print(y)
# print(y.shape)  # (2391, 1, 1)

x_pred = dataset[-1:,:,:]
# print(x_pred)
# print(x_pred.shape) # (1, 6, 6)


# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,\
    shuffle=True, random_state=31)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, \
    train_size=0.8, shuffle=True, random_state=31)
# print(x_train.shape)        # (1529, 6, 6)
# print(x_test.shape)         # (479, 6, 6)
# print(x_validation.shape)   # (383, 6, 6)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
y_validation = y_validation.reshape(y_validation.shape[0],1)
# print(y_train.shape)        # (1529, 1)
# print(y_test.shape)         # (479, 1)
# print(y_validation.shape)   # (383, 1)

# MinMaxscaler를 하기 위해서 2차원으로 바꿔준다.
x_train = x_train.reshape(x_train.shape[0],size*col)
x_test = x_test.reshape(x_test.shape[0],size*col)
x_validation = x_validation.reshape(x_validation.shape[0],size*col)
x_pred = x_pred.reshape(x_pred.shape[0],size*col)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0],size,col)
x_test = x_test.reshape(x_test.shape[0],size,col)
x_validation = x_validation.reshape(x_validation.shape[0],size,col)
x_pred= x_pred.reshape(x_pred.shape[0], size,col)

# print(x_train.shape)        # (1529, 6, 6)
# print(x_test.shape)         # (479, 6, 6)
# print(x_validation.shape)   # (383, 6, 6)
# print(x_pred.shape)         # (1, 6, 6)

#2. Modeling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D


# model = load_model('./samsung/cp/samsung_c_day15_157-530667.9375.h5')
# model = load_model('./samsung/cp/samsung_c_day15_173-571237.1250.h5')
# model = load_model('./samsung/cp/samsung_l_day15_427-479009.1250.h5')
model = load_model('./samsung/samsung_l_day15_925-1243966.6250.h5')


result = model.evaluate(x_test, y_test, batch_size=size)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

predict = model.predict(x_pred)
print("1월 15일 삼성주가 예측 : ", predict)

######################## Conv1D ########################
# loss :  517567.0625
# mae :  497.7959899902344
# RMSE :  517567.8
# R2 :  0.9969870033979905
# 1월 15일 삼성주가 예측 :  [[88977.03]]

# loss :  549417.4375
# mae :  523.0789184570312
# RMSE :  549419.8
# R2 :  0.9968015783158997
# 1월 15일 삼성주가 예측 :  [[89376.664]] ***
######################## LSTM ########################
# loss :  555647.625
# mae :  549.9356689453125
# RMSE :  555647.7
# R2 :  0.9967653228325574
# 1월 15일 삼성주가 예측 :  [[92908.24]]

# loss :  1263672.625
# mae :  833.64599609375
# RMSE :  1263672.1
# R2 :  0.992643592124265
# 1월 15일 삼성주가 예측 :  [[96051.74]]