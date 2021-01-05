# 다중 분류

import numpy as np
from sklearn.datasets import load_iris

#1. DATA

dataset = load_iris()
x = dataset.data
y = dataset.target

# print(dataset.DESCR)
# print(dataset.feature_names)

print(x[:5])
print(y)        # 0, 1, 2 >> 다중 분류
print(x.shape)  #(150, 4)
print(y.shape)  #(150, )

# x-preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, \
    train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# y-preprocessing (1)
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# print(y_train.shape)    # (120, 3)
# print(y_test.shape)     # (30, 3)

# y-preprocessing (2)
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

encoder = OneHotEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()

print(y_train.shape)    # (120, 3)
print(y_test.shape)     # (30, 3)


#2. Modeling
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
output1 = Dense(3, activation='softmax')(dense1)
model = Model(inputs = input1, outputs = output1)
# model.summary()

#3. Compile, Train
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10,mode='min')
model.fit(x_train, y_train, epochs = 10, batch_size=5,\
    validation_split = 0.2, verbose=1, callbacks=[early_stopping])

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=5)
print("loss : ", loss)
print('acc : ', acc)

y_pred = model.predict(x_test[-5:-1])
print("y_pred : \n", y_pred)

print("y_pred_012 :\n", np.argmax(y_pred, axis=1))