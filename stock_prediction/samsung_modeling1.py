# 다음 날 삼성의 '종가'를 예측한다.

import numpy as np

data = np.load('./stock_prediction/samsung_slicing_data1.npy')

# size : 며칠씩 자를 것인지
# col : 열의 개수

def split_x(seq, size, col) :
    dataset = []  
    for i in range(len(seq) - size + 1) :
        subset = seq[i:(i+size),0:col].astype('float32')
        dataset.append(subset)
    # print(type(dataset))
    return np.array(dataset)

dataset = split_x(data,5, 6)
# print(dataset)
# print(dataset.shape) # (2393, 5, 6)

#1. DATA
x = dataset[:-1,:,:7]
# print(x)
print(x.shape)  # (2392, 5, 6)

y = dataset[1:,:1,-1:]
# print(y)
print(y.shape)  # (2392, 1, 1)

x_pred = dataset[-1:,:,:]
# print(x_pred)
print(x_pred.shape) # (1, 5, 6)

# preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,\
    shuffle=True, random_state=166)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, \
    train_size=0.8, shuffle=True, random_state=166)

print(x_train.shape)    # (1530, 5, 6)
print(x_test.shape)     # (479, 5, 6)

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
print(y_train.shape)    # (1530, 1)
print(y_test.shape)     # (479, 1, 1)

# MinMaxscaler를 하기 위해서 2차원으로 바꿔준다.
x_train = x_train.reshape(x_train.shape[0],30)
x_test = x_test.reshape(x_test.shape[0],30)
x_validation = x_validation.reshape(x_validation.shape[0],30)
x_pred = x_pred.reshape(x_pred.shape[0],30)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_validation = scaler.transform(x_validation)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0],5, 6)
x_test = x_test.reshape(x_test.shape[0],5, 6)
x_validation = x_validation.reshape(x_validation.shape[0], 5, 6)
x_pred= x_pred.reshape(x_pred.shape[0], 5, 6)

print(x_train.shape)    # (1530, 5, 6)
print(x_test.shape)     # (479, 5, 6)
print(x_validation.shape) # (383, 5, 6)

#2. Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(512, activation='relu', \
    input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
savepath = './stock_prediction/samsung_day14_{epoch:02d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss',patience=10,mode='min')
cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=6, \
    validation_data=(x_validation, y_validation),verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
result = model.evaluate(x_test, y_test, batch_size=6)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred = model.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))
print("RMSE : ", RMSE(y_test, y_pred))

r2 = r2_score(y_test, y_pred)
print("R2 : ", r2)

predict = model.predict(x_pred)
print("predict : ", predict)


# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))  # 판 사이즈

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')   # label=' ' >> legend에서 설정한 위치에 라벨이 표시된다.
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')   # loc 를 명시하지 않으면 그래프가 비어있는 지역에 자동으로 위치한다.

plt.show()

# loss :  9671933.0
# mae :  2874.96875
# RMSE :  9671934.0
# R2 :  0.9391480643231802
# predict :  [[78046.78]]