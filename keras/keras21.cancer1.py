# breask cancer data
# classification model (이진분류 모델)
# loss = 'binary_crossentropy', activation = 'sigmoid', metrics=['acc']

# 실습1. acc : 0.985 이상 올릴 것
# 실습2. 일부데이터의 원래 데이터와 y_predict 데이터 비교
# 실습3. 결과가 0과 1 둘 중 하나로 나오게 코딩 할 것

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
datasets = load_breast_cancer()

# print(datasets.DESCR)           # (569, 30)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

print(x.shape)  #(569, 30) , input_dim = 30
print(y.shape)  #(568, ) # 유방암에 걸렸는지 안 걸렸는지 , output = 1

# print(x[:5])
# print(y)        # 0 or 1 >> classification (이진분류)

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(120, activation='relu', input_shape=(30,)))
model.add(Dense(90, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='sigmoid')) #(히든 레이어 없어도 된다.) # sigmoid : 마지막 결과값은 0과 1사이로 나와야 한다.

# 함수형 왜 안되는 건지 다시 확인하기
# input1 = Input(shape=(30,))
# dense1 = Dense(30, activation='relu')(input1)
# dense1 = Dense(30, activation='relu')(dense1)
# dense1 = Dense(30, activation='relu')(dense1)
# dense1 = Dense(30, activation='relu')(dense1)
# output1 = Dense(30, activation='sigmoid')(dense1)
# model = Model(inputs = input1, outputs = output1)

#3. Compile, Train
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])  # acc == accuracy
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # mse == mean_squared_error (풀 네임으로 적어도 된다.)

# from tensorflow.keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 
# model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=1,callbacks=[early_stopping])

model.fit(x_train, y_train, epochs=450, batch_size=15, validation_split=0.2, verbose=1)

#4. Evalutate Predcit
loss, acc = model.evaluate(x_test, y_test,batch_size=15)
print("loss : ",loss)
print("accuracy : ", acc)

y_predict = model.predict(x_test[-5:-1])

y_binary = list()           # y_preedict 값을 0과 1로 변환한 데이터 저장할 리스트
for i in y_predict :
    if i >= 0.5 :
        y_binary.append(1)  # y_preedict >= 0.5 이면 1
    else : 
        y_binary.append(0)  # y_preedict < 0.5 이면 0

print("y_test_data : ", y_test[-5 : -1])
print("y_predict :\n", y_predict)
print("result :", y_binary)


# loss :  0.6069278120994568
# accuracy :  0.9736841917037964
# y_test_data :  [1 0 1 1]
# y_predict :
#  [[1.00000000e+00]
#  [1.30554526e-11]
#  [1.00000000e+00]
#  [1.00000000e+00]]
# result : [1, 0, 1, 1]