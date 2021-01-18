# Conv1D
# 1월 19일의 삼성전자 시가를 예측한다.

import numpy as np

#1. DATA
# x1 >> x_sam : 삼성 데이터
x_train_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[0]
x_test_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[1]
x_val_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[2]
y_train_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[3]
y_test_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[4]
y_val_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[5]
x_pred_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[6]

# print(x_train_sam.shape)        # (689, 6, 6)
# print(x_test_sam.shape)         # (216, 6, 6)
# print(x_val_sam.shape)          # (173, 6, 6)
# print(y_train_sam.shape)        # (689, 2)
# print(y_test_sam.shape)         # (216, 2)
# print(x_pred_sam.shape)         # (1, 6, 6)

# x2 >> x_kod : KODEX 코스닥150 선물 인버스 데이터
x_train_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[0]
x_test_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[1]
x_val_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[2]
x_pred_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[3]

# print(x_train_kod.shape)        # (689, 6, 6)
# print(x_test_kod.shape)         # (216, 6, 6)
# print(x_val_kod.shape)          # (173, 6, 6)
# print(x_pred_kod.shape)         # (1, 6, 6)

size = 6
col = 6

#2. Modeling 
#3. Compile, Train
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPool1D

# Conv1D
model = load_model('./samsung/cp/ensemble_c_day19_92-891383.3125.h5')

#4. Evaluate, Predict
result = model.evaluate([x_test_sam,x_test_kod], y_test_sam, batch_size=size)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred_sam = model.predict([x_test_sam, x_test_kod])

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))

RMSE_sam = RMSE(y_test_sam, y_pred_sam)
print("RMSE : ", RMSE_sam)

r2_sam = r2_score(y_test_sam, y_pred_sam)
print("R2 : ", r2_sam)

predict = model.predict([x_pred_sam,x_pred_kod])
print("predict : ", predict)

print("C_1월 18일 삼성전자 시가 : ", predict[0,0:1])
print("C_1월 19일 삼성전자 시가 : ", predict[0,1:])

# loss :  774932.25
# mae :  619.716064453125
# RMSE :  774932.2
# R2 :  0.9895120431507425
# predict :  [[87920.86 88118.83]]
# C_1월 18일 삼성전자 시가 :  [87920.86]
# C_1월 19일 삼성전자 시가 :  [88118.83]