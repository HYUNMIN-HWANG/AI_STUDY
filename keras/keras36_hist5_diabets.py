# hist를 이용하여 그래프를 그리시오
# loss, val_loss

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

#1. DATA
x = dataset.data
y = dataset.target

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

# 전처리 과정

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)


#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=10, activation='linear'))
model.add(Dense(100, activation='linear'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='min') 

hist = model.fit(x_train, y_train, epochs=5000, batch_size=5, validation_data=(x_validation, y_validation), verbose=1,callbacks=[early_stopping] )


# 그래프 그리기
import matplotlib.pyplot as plt 

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train loss','val loss'])
plt.show()

#4. Evaluate, Predict
loss, mae = model.evaluate(x_test, y_test, batch_size=5)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict) :
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

# R2
from sklearn.metrics import r2_score
r2 = r2_score (y_test, y_predict)
print("R2 : ", r2)

# Early Stopping
# loss :  2305.467529296875
# mae :  39.62618637084961
# RMSE :  48.01528725236817
# R2 :  0.5981309295781687

