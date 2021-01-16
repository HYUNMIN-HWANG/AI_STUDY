import numpy as np

#1. DATA
# x1 >> x_sam : 삼성 데이터
x_train_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[0]
x_test_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[1]
x_val_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[2]
y_train_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[3]
y_test_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[4]
y_val_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[5]
x_pred_sam = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/samsung_slicing_data5.npy',allow_pickle=True)[6]

# print(x_train_sam.shape)        # (689, 6, 6)
# print(x_test_sam.shape)         # (216, 6, 6)
# print(x_val_sam.shape)          # (173, 6, 6)
# print(y_train_sam.shape)        # (689, 2)
# print(y_test_sam.shape)         # (216, 2)
# print(x_pred_sam.shape)         # (1, 6, 6)

# x2 >> x_kod : KODEX 코스닥150 선물 인버스 데이터
x_train_kod = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/kodex_slicing_data1.npy',allow_pickle=True)[0]
x_test_kod = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/kodex_slicing_data1.npy',allow_pickle=True)[1]
x_val_kod = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/kodex_slicing_data1.npy',allow_pickle=True)[2]
x_pred_kod = np.load('/content/drive/My Drive/인공지능 과정/stock_prediction/kodex_slicing_data1.npy',allow_pickle=True)[3]

# print(x_train_kod.shape)        # (689, 6, 6)
# print(x_test_kod.shape)         # (216, 6, 6)
# print(x_val_kod.shape)          # (173, 6, 6)
# print(x_pred_kod.shape)         # (1, 6, 6)

size = 6
col = 6

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, concatenate

# input1 = samsung model
input1 = Input(shape=(x_train_sam.shape[1],x_train_sam.shape[2]))
lstm1 = LSTM(512, activation='relu')(input1)
dense1 = Dense(256, activation='relu')(lstm1)
dense1 = Dense(128, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)

# input2 = kodex model
input2 = Input(shape=(x_train_kod.shape[1],x_train_kod.shape[2]))
lstm2 = LSTM(512, activation='relu')(input2)
dense2 = Dense(256, activation='relu')(lstm2)
dense2 = Dense(128, activation='relu')(dense2)
dense2 = Dense(64, activation='relu')(dense2)

# concatenate
merge1 = concatenate([dense1, dense2])
dense3 = Dense(128, activation='relu')(merge1)
dense3 = Dense(64, activation='relu')(dense3)
dense3 = Dense(32)(dense3)

# output
output1 = Dense(2)(dense3)   # y1 :output =  2 (마지막 아웃풋)

model = Model(inputs=[input1,input2], outputs=output1)

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
savepath = '/content/drive/My Drive/인공지능 과정/stock_prediction/cp/ensemble_l_day19_{epoch:02d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss',patience=60,mode='min')
cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit([x_train_sam,x_train_kod], y_train_sam, epochs=2000, batch_size=size, \
    validation_data=([x_val_sam,x_val_kod], y_val_sam),verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
result = model.evaluate([x_test_sam,x_test_kod], y_test_sam, batch_size=size)
print("loss : ", result[0])
print("mae : ", result[1])

y_pred_sam = model.predict([x_test_sam, x_test_kod])

from sklearn.metrics import mean_squared_error, r2_score
def RMSE (y_test, y_pred) :
    return np.array(mean_squared_error(y_test, y_pred))

RMSE_sam = RMSE(y_test_sam, y_pred_sam)
print("RMSE : ", RMSE_sam)

r2_sam = r2_score(y_test_sam, y_pred_sam)
print("R2 : ", r2_sam)

predict = model.predict([x_pred_sam,x_pred_kod])
print("L_1월 18일, 19일 : ", predict)

print("L_1월 19일 삼성전자 시가 : ", predict[0,1:])

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(25,15))  # 판 사이즈

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')   
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')   

plt.show()

# loss :  1662954.125
# mae :  947.4702758789062
# RMSE :  1662954.6
# R2 :  0.9773110802555567
# L_1월 18일, 19일 :  [[84319.914 84288.87 ]]
# L_1월 19일 삼성전자 시가 :  [84288.87]

# loss :  2425706.75
# mae :  1266.14208984375
# RMSE :  2425708.5
# R2 :  0.966884453734451
# L_1월 18일, 19일 :  [[88665.195 88758.56 ]]
# L_1월 19일 삼성전자 시가 :  [88758.56]

# loss :  2498246.0
# mae :  1287.2281494140625
# RMSE :  2498245.5
# R2 :  0.9659227999713391
# L_1월 18일, 19일 :  [[91111.71  91309.516]]
# L_1월 19일 삼성전자 시가 :  [91309.516]

# loss :  1078221.125
# mae :  756.5844116210938
# RMSE :  1078222.2
# R2 :  0.9853445469774161
# L_1월 18일, 19일 :  [[87678.72 87115.57]]
# L_1월 19일 삼성전자 시가 :  [87115.57]
