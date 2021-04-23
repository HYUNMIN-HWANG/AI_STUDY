# 다음 날 삼성의 '종가'를 예측한다.

import numpy as np

#1. 전체 DATASET
x_train = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[0]
x_test = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[1]
x_validation = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[2]
y_train = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[3]
y_test = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[4]
y_validation = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[5]
x_pred = np.load('./samsung/samsung_slicing_data4.npy',allow_pickle=True)[6]
# print(x_train.shape)        # (1530, 6, 6)
# print(x_test.shape)         # (479, 6, 6)
# print(x_validation.shape)   # (383, 6, 6)
# print(y_train.shape)        # (1530, 1)
# print(y_test.shape)         # (479, 1)
# print(x_pred.shape)         # (1, 6, 6)

size = 6
col = 6

#2. Modeling 
#3. Compile, Train
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D

model = load_model('./samsung/cp/samsung_c_day15_157-530667.9375.h5')

#4. Evaluate, Predict
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

# Conv1D
# loss :  544228.3125
# mae :  512.66650390625
# RMSE :  544228.44
# R2 :  0.9967006791672697
# 1월 15일 삼성주가 예측 :  [[88963.88]]