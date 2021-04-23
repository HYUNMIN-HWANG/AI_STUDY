# 다음 날 삼성의 '종가'를 예측한다.

import numpy as np

# 전체 DATASET
data = np.load('./samsung/samsung_slicing_data2.npy')
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

# model = load_model('./samsung/samsung_c_day14_133-638980.4375.h5')
# model = load_model('./samsung/samsung_l_day14_17-2424388.5000.h5')
model = load_model('./samsung/samsung_l_day14_432-679153.7500.h5')


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
print("1월 14일 삼성주가 예측 : ", predict)

# loss :  578553.1875
# mae :  552.487060546875
# RMSE :  578553.25
# R2 :  0.9967384613575851
# 1월 14일 삼성주가 예측 :  [[92838.805]]

# loss :  2674416.25
# mae :  1198.0443115234375
# RMSE :  2674416.2
# R2 :  0.9849232346434873
# 1월 14일 삼성주가 예측 :  [[89744.54]]