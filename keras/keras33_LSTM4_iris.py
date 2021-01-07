# sklearn
# LSTM으로 모델링 (Dense 와 성능 비교)
# 다중분류 

# y데이터 전처리 >>>> 원핫인코딩 (1) from tensorflow.keras.utils import to_categorical
# activation = 'softmax', loss='categorical_crossentropy'

import numpy as np
from sklearn.datasets import load_iris

#1. DATA

dataset = load_iris()
x = dataset.data 
y = dataset.target 

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )

# x값 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=133)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)    # (120, 4)
print(x_test.shape)     # (30, 4)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)    # (120, 4, 1)
print(x_test.shape)     # (30, 4, 1)

# 다중 분류일 때, y값 전처리 One hot Encoding >> tensorflow.keras 사용
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
# print(y_train)
# print(y_test)
print(y_train.shape)    # (120, 3) >>> output = 3
print(y_test.shape)     # (30, 3)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(16, activation='relu', input_shape=(4,1)))   #input = 4
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))                  
            # output = 3 (다중분류모델에서는 분류하는 수만큼 노드개수를 정한다.)                           
            # softmax : 마지막 노드를 다 합치면 1이 된다. > 그 중에서 가장 큰 값이 선택된다.
model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])  # acc == accuracy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
model.fit(x_train, y_train, epochs=200, batch_size=5, validation_split=0.2, verbose=1, callbacks=[early_stopping])


#4. Evaluate, Predict
loss, acc, mae  = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ", loss)
print("accuracy : ", acc)
print("mae : ", mae)

y_predict = model.predict(x_test[-5:-1])
print("y_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)   
# 결과가 0,1,2가 아닌 소수가 나온다. >> softmax : 분류하고자 하는 숫자의 개수만큼 값이 분리된다. 다 합하면 1
# 원하는 결과가 나오도록 0,1,2로 정제해야 함 >> argmax

# y값 중에서 가장 큰 값이 있는 위치를 반환해줌
# argmax - 0은 열(column), 1은 행(row), 2는 면(page, 행열)
print(np.argmax(y_predict,axis=1))


# Dense
# loss :  0.05020342767238617
# accuracy :  1.0
# mae :  0.0286836139857769
# y_data :
#  [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# y_predict :
#  [[2.2208088e-15 2.0895617e-04 9.9979109e-01]
#  [9.9998808e-01 1.1910735e-05 1.0679650e-27]
#  [9.8897481e-01 1.1025138e-02 1.3339521e-15]
#  [1.0321673e-08 3.9325234e-01 6.0674769e-01]]
# [2 0 0 2]

# LSTM
# loss :  0.18687547743320465
# accuracy :  1.0
# mae :  0.1052408218383789
# y_data :
#  [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# y_predict :
#  [[3.3189059e-15 3.7193631e-03 9.9628061e-01]
#  [9.8383743e-01 1.6072018e-02 9.0519665e-05]
#  [9.6981603e-01 2.8719692e-02 1.4643057e-03]
#  [7.5999578e-06 4.9553013e-01 5.0446224e-01]]
# [2 0 0 2]