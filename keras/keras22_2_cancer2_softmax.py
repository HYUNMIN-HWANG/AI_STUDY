# keras21_cancer1.py를 다중분류로 코딩하시오

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
datasets = load_breast_cancer()

# print(datasets.DESCR)           # (569, 30)
# print(datasets.feature_names)

x = datasets.data
y = datasets.target

# print(x.shape)  #(569, 30) , input_dim = 30
# print(y.shape)  #(568, ) # 유방암에 걸렸는지 안 걸렸는지 , output = 1

# print(x[:5])
# print(y)        # 0 or 1 >> classification (이진분류)

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (455, 2)
print(y_test.shape)     # (114, 2)

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(150, activation='relu', input_shape=(30,)))
model.add(Dense(90, activation='relu'))
model.add(Dense(90, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax')) # sigmoid >> softmax


#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # acc == accuracy
model.fit(x_train, y_train, epochs=450, batch_size=15, validation_split=0.2, verbose=1)

#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test,batch_size=15)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)

y_predict = model.predict(x_test[-5:-1])

print("y_test_data :\n", y_test[-5 : -1])
print("y_predict :\n", y_predict)

print("result : ", np.argmax(y_predict,axis=1))

# loss :  0.5559095740318298
# accuracy :  0.9736841917037964
# mae :  0.041522957384586334
# y_test_data :
#  [[0. 1.]
#  [1. 0.]
#  [0. 1.]
#  [0. 1.]]
# y_predict :
#  [[3.6245937e-12 1.0000000e+00]
#  [1.0000000e+00 5.6693203e-15]
#  [4.5523027e-10 1.0000000e+00]
#  [3.3059730e-10 1.0000000e+00]]
# result :  [1 0 1 1]