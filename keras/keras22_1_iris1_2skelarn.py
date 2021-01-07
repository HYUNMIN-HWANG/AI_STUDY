# 다중 분류
# activation = 'softmax'
# y데이터 전처리 >>>> 원핫인코딩 (2) from sklearn.preprocessing import OneHotEncoder
# loss='categorical_crossentropy'
# argmax : 가장 큰 값을 찾아준다.

import numpy as np
from sklearn.datasets import load_iris

#1. DATA
# x, y, = load_iris(return_X_y=True) # x와 y를 분리하는 방법
# 아래 3 줄과 동일함
dataset = load_iris()
x = dataset.data 
y = dataset.target 
# print(dataset.DESCR)
# print(dataset.feature_names)

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )
# print(x[:5])
# print(y)        # 나올 수 있는 경우의 수 3가지 : 0 , 1 , 2 (50개 씩) >>> 원핫인코딩해야 함

# x값 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 다중 분류일 때, y값 전처리 One hot Encoding (1) tensorflow.keras 사용
# from tensorflow.keras.utils import to_categorical
# from keras.utils.np_utils import to_categorical # 옛날 버전

# 다중 분류일 때, y값 전처리 One hot Encoding (2) sklearn 사용
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()  #toarray() : list 를 array로 바꿔준다.
y_test = encoder.transform(y_test).toarray()

print(y_train.shape)    # (120, 3)  #output = 3
print(y_test.shape)     # (30, 3)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(4,)))   #input = 4
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))                   #output = 3           
            # output = 3 (다중분류모델에서는 분류하는 수만큼 노드개수를 정한다.)
            # 마지막 노드를 다 합치면 1이 된다. > 그 중에서 가장 큰 값이 선택된다.

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc','mae'])  # acc == accuracy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min') 
model.fit(x_train, y_train, epochs=3000, batch_size=5, validation_split=0.2, verbose=1,callbacks=[early_stopping])

#4. Evaluate, Predict
loss, acc,mae  = model.evaluate(x_test, y_test,batch_size=5)
print("loss : ", loss)
print("accuracy : ", acc)
print("mae : ", mae)

y_predict = model.predict(x_test[-5:-1])
print("y_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)

# y값 중에서 가장 큰 값을 1로 바꾼다. : argmax - 0은 열(column), 1은 행(row), 2는 면(page, 행열)
print("result : " ,np.argmax(y_predict,axis=1))

# loss :  0.052693843841552734
# accuracy :  1.0
# mae :  0.030473845079541206
# y_data :
#  [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]]
# y_predict :
#  [[2.5443098e-12 1.0436538e-05 9.9998951e-01]
#  [9.9998975e-01 1.0210336e-05 2.1660636e-09]
#  [9.9858296e-01 1.4131406e-03 3.8798580e-06]
#  [5.0007893e-06 2.7610904e-01 7.2388601e-01]]
# result : 
#   [2 0 0 2]