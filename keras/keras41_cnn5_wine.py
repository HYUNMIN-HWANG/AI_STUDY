# CNN 으로 구성
# 2차원 데이터를 4차원으로 늘려서 하시오.

import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()

#1. DATA

x = dataset.data
y = dataset.target

# print(x)        # preprocessing 해야 함
# print(y)        # 0, 1, 2 >> 다중분류
# print(x.shape)  # (178, 13)
# print(y.shape)  # (178, )

# x > preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, shuffle=True, random_state=55)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],13, 1, 1)
x_test = x_test.reshape(x_test.shape[0],13, 1, 1)

print(x_train.shape)    # (160, 13, 1, 1)
print(x_test.shape)     # (18, 13, 1, 1)

# y > preprocessing
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)
print(y_train.shape)    # (160, 3)
print(y_test.shape)     # (18, 3)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

model = Sequential()
model.add(Conv2D(filters=65, kernel_size=(2,1),padding='same',\
                input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(Dropout(0.1))
model.add(Conv2D(filters=26, kernel_size=(2,1)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))                   # output = 3

model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mae'])  # 다중 분류 : categorical_crossentropy 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min') 
model.fit(x_train, y_train, epochs=45, batch_size=13, validation_split=0.1, verbose=1,callbacks=[early_stopping])

#4. Evalutate Predcit
loss, acc, mae = model.evaluate(x_test, y_test, batch_size=13)
print("loss : ",loss)
print("accuracy : ", acc)
print("mae : ", mae)

print("y_test :",np.argmax(y_test[-5 : -1],axis=1))

y_predict = model.predict(x_test[-5:-1])
print("y_predict :", np.argmax(y_predict,axis=1))

# Dropout
# loss :  0.028161056339740753
# accuracy :  1.0
# mae :  0.01674073189496994
# y_test_data :
#  [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]]
# y_predict :
#  [[9.9903333e-01 9.0184662e-04 6.4793065e-05]
#  [4.2299906e-04 9.9957305e-01 3.8741309e-06]
#  [2.3380478e-07 2.4176154e-07 9.9999952e-01]
#  [9.9871445e-01 1.2303371e-03 5.5225402e-05]]
# result :  [0 1 2 0]

# loss :  0.02343757078051567
# accuracy :  1.0
# mae :  0.014686604030430317
# y_test : [0 1 2 0]
# y_predict : [0 1 2 0]