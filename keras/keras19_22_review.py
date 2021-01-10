import numpy as np
# from sklearn.datasets import load_diabetes
# from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

# dataset = load_diabetes()
# dataset = load_breast_cancer()
dataset = load_iris()

x = dataset.data
y = dataset.target

#1. DATA
print(x.shape)  # (150, 4)
print(y.shape)  # (150,)

print(x[:5])    # 전처리 필요함
print(y)        # 0 or 1 or 2 > 다중분류

# x > preprocessing

# x = x / 0.198787989657293

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=55)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=55)

print(x_train.shape)    # (96, 4)
print(x_test.shape)     # (30, 4)

# Minmaxscaler
print(np.min(x_train))  # 최솟값 : 0.1
print(np.max(x_train))  # 최댓값 : 7.9

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print(np.min(x_train))  # 최솟값 : 0.0
print(np.max(x_train))  # 최댓값 : 1.0

# y > preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# from sklearn.preprocessing import OneHotEncoder
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# y_val = y_val.reshape(-1,1)

# encoder = OneHotEncoder()
# encoder.fit(y_train)
# encoder.fit(y_test)
# encoder.fit(y_val)

# y_train = encoder.transform(y_train).toarray()
# y_test = encoder.transform(y_test).toarray()
# y_val = encoder.transform(y_val).toarray()

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(4,)) # input = 4
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
output1 = Dense(3, activation='softmax')(dense1) # output = 3

model = Model(inputs=input1, outputs=output1)

model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='categorical_crossentropy',patience=10,mode='min')
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=2, validation_data=(x_val, y_val), verbose=2, callbacks=[es])

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=2)
print("loss : ", loss)
print("mae : ", mae)

print("y_test : ",np.argmax(y_test[-5:-1],axis=1))
y_pred = model.predict(x_test[-5:-1])
print("y_pred : ",np.argmax(y_pred, axis=1))

# RMSE
# from sklearn.metrics import mean_squared_error, r2_score

# def RMSE (y_test, y_pred) :
#     return np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE : ", RMSE(y_test, y_pred))

# R2
# r2 = r2_score(y_test, y_pred)
# print("R2 : ", r2)

