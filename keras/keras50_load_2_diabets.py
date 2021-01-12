# 저장한 numpy 불러오기 : np.load

import numpy as np
x_data = np.load('../data/npy/diabets_x.npy')
y_data = np.load('../data/npy/diabets_y.npy')

# print(x_data)
# print(y_data)
print(x_data.shape) # (442, 10)
print(y_data.shape) # (442,)

# =========================== 모델을 완성하시오 ===========================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True)
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
model.add(Dense(100, input_dim=10, activation='relu'))
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
modelpath='../data/modelcheckpoint/k46_5_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'
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


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))  # 판 사이즈 (가로 10, 세로 6)

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

# plt.title('손실비용') # 과제 : 한글 깨짐 오류 해결할 것
plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# Dense
# loss :  2244.21240234375
# mae :  38.237037658691406
# RMSE :  47.373117107794975
# R2 :  0.49755893626202496

# ModelCheckPoint
# loss :  3172.412109375
# mae :  45.06081771850586
# RMSE :  56.32417273888865
# R2 :  0.45417605302059283