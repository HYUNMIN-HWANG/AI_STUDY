# 실습 : 보스턴 집값 예측하기

# 다 : 1 mlp 모델
# 전처리 전

import numpy as np

from sklearn.datasets import load_boston
dataset = load_boston()

#1. DATA
x = dataset.data
y = dataset.target

print(x.shape)  #(506, 13) input = 13
print(y.shape)  #(506, )   output = 1

print( np.min(x), np.max(x))    # 0.0 ~ 711.0
print(dataset.feature_names)    
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, \
    train_size=0.8, shuffle=True, random_state=66)

#2. Modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbos =1)

#4. Evaluate, Predict
loss. mae = model.evaluate(x_test, y_test, batch_size=10)
print("loss : ", loss)
print("mae : ", mae)

y_predcit = model.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predcit) : 
    return  np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE)


# R2
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print(r2)