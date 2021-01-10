# LSTM

import numpy as np

from sklearn.datasets import load_boston, load_diabetes, load_breast_cancer, load_iris, load_wine
# dataset = load_boston()
# dataset = load_diabetes()
# dataset = load_breast_cancer()
# dataset = load_iris()
dataset = load_wine()

#1. DATA
x = dataset.data
y = dataset.target
print(x.shape)  # (178, 13)
print(y.shape)  # (178,)

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=112)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

print(x_train.shape)    # (142, 13, 1)
print(x_test.shape)     # (36, 13, 1)

# print(y)                # 0 or 1 or 2 > 다중분류

# y > preprocessing
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)    # (142, 3)
print(y_test.shape)     # (36, 3)

#2. Modeling
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
# model = load_model("./STUDY/save_keras33_40_review.h5")

model = Sequential()
model.add(LSTM(40, activation='relu',input_shape=(13,1)))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dense(20, name='new1'))
model.add(Dense(3, activation='softmax'))

model.summary()
# model.save("./STUDY/save_keras33_40_review.h5")

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss',patience=8,mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=13,validation_split=0.2, verbose=2, callbacks=[es])

# print(hist)
# print(hist.history.keys())
# print(hist.history['loss'])

#4. Evaluate, Predict
loss, acc = model.evaluate(x_test, y_test, batch_size=13)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(x_test[-5:-1])
print("y_pred : ")
# print(np.where(y_pred>0.5,1,0))
print(np.argmax(y_pred,axis=1))

# Graph
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['loss','val loss','acc','val acc'])
plt.show()
# RMSE
# from sklearn.metrics import mean_squared_error, r2_score
# def RMSE (y_pred, y_test) :
#     return np.sqrt( mean_squared_error(y_pred, y_test))
# print("RMSE : ", RMSE(y_pred, y_test))

# # R2 
# r2 = r2_score(y_pred, y_test)
# print("R2 : ", r2)
