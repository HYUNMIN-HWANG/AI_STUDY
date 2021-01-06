# LSTM 두 개가 들어간 ensemble 다 : 1 모델 (예측값을 85 근사치로 만들어라)

import numpy as np

#1. DATA
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
                [5,6,7],[6,7,8],[7,8,9],[8,9,10],
                [9,10,11],[10,11,12],
                [20,30,40],[30,40,50],[40,50,60]])
x2 = np.array([[10,11,12],[20,30,40],[30,40,50],[40,50,60],
                [50,60,70],[60,70,80],[70,80,90],[80,90,100],
                [90,100,110],[100,110,120],
                [2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) 
x1_predict = np.array([55,65,75])
x2_predict = np.array([65,75,85])

# print(x1.shape) #(13, 3)
# print(x2.shape) #(13, 3)
# print(y.shape)  #(13, )
# print(x1_predict.shape) # (3,)
# print(x2_predict.shape) # (3,)

x1_predict = x1_predict.reshape(1,3)
x2_predict = x2_predict.reshape(1,3)

# 전처리

#2. Modeling

#3. Compile, Train

#4. Evaluate, Predict

