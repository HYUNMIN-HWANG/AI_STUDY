# Dropout
# model.add(Dropout(0.2))
# Dropout 하기 전과 성능 비교

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

#1. DATA
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])

print(x.shape, y.shape)         #(442, 10) (442,) input = 10, output = 1

print(np.max(x), np.min(y))     # 0.198787989657293 25.0  ---> 전처리 해야 함
print(np.max(x), np.min(x))     # 0.198787989657293 -0.137767225690012
print(dataset.feature_names)    # 10 column
                                # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

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
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(200, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#3. Compile, Train
model.compile(loss='mse', optimizer='adam',metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath='./modelCheckpoint/k46_5_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='loss', patience=10, mode='min') 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True, mode='auto')

hist = model.fit(x_train, y_train, epochs=5000, batch_size=5, validation_data=(x_validation, y_validation), verbose=1,callbacks=[es,cp] )

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



# Dense
# loss :  2244.21240234375
# mae :  38.237037658691406
# RMSE :  47.373117107794975
# R2 :  0.49755893626202496

# ModelCheckPoint
# loss :  2729.91552734375
# mae :  42.15181350708008
# RMSE :  52.248596086172874
# R2 :  0.5510438539208138