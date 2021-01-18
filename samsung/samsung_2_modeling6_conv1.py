import numpy as np

#1. DATA
# x1 >> x_sam : 삼성 데이터
x_train_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[0]
x_test_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[1]
x_val_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[2]
y_train_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[3]
y_test_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[4]
y_val_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[5]
x_pred_sam = np.load('./samsung/samsung_slicing_data5.npy',allow_pickle=True)[6]

print(x_train_sam.shape)        # (689, 6, 6)
print(x_test_sam.shape)         # (216, 6, 6)
print(x_val_sam.shape)          # (173, 6, 6)
print(y_train_sam.shape)        # (689, 2)
print(y_test_sam.shape)         # (216, 2)
print(x_pred_sam.shape)         # (1, 6, 6)

# x2 >> x_kod : KODEX 코스닥150 선물 인버스 데이터
x_train_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[0]
x_test_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[1]
x_val_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[2]
x_pred_kod = np.load('./samsung/kodex_slicing_data1.npy',allow_pickle=True)[3]

print(x_train_kod.shape)        # (689, 6, 6)
print(x_test_kod.shape)         # (216, 6, 6)
print(x_val_kod.shape)          # (173, 6, 6)
print(x_pred_kod.shape)         # (1, 6, 6)

size = 6
col = 6

#2. Modeling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Input, Dropout, Conv1D, Flatten, MaxPool1D, concatenate

# input1 = samsung model
input1 = Input(shape=(x_train_sam.shape[1],x_train_sam.shape[2]))
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input1)
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv1)

conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
conv1 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)

conv1 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(conv1)

flat1 = Flatten()(conv1)

# input2 = kodex model
input2 = Input(shape=(x_train_kod.shape[1],x_train_kod.shape[2]))
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input2)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv2)
conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(conv2)

conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv2)
conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv2)

conv2 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(conv2)

flat2 = Flatten()(conv2)

# concatenate
merge1 = concatenate([flat1, flat2])
dense3 = Dense(128, activation='relu')(merge1)
dense3 = Dense(64, activation='relu')(dense3)
dense3 = Dense(32, activation='relu')(dense3)

# output
output1 = Dense(2)(dense3)   # y1 :output =  2 (마지막 아웃풋)

model = Model(inputs=[input1,input2], outputs=output1)

# model.summary()

#3. Compile, Train
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
savepath = '/content/drive/My Drive/인공지능 과정/stock_prediction/cp/ensemble_c_day19_{epoch:02d}-{val_loss:.4f}.h5'

es = EarlyStopping(monitor='val_loss',patience=80,mode='min')
cp = ModelCheckpoint(filepath=savepath, monitor='val_loss',save_best_only=True,mode='min')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
hist = model.fit([x_train_sam,x_train_kod], y_train_sam, epochs=2000, batch_size=6, \
    validation_data=([x_val_sam,x_val_kod], y_val_sam),verbose=1,callbacks=[es, cp])

#4. Evaluate, Predict
result = model.evaluate([x_test_sam,x_test_kod], y_test_sam, batch_size=6)
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
print("C_1월 18일, 19일 : ", predict)

print("C_1월 19일 삼성전자 시가 : ", predict[0,1:])


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


# loss :  847671.0625
# mae :  653.3145751953125
# RMSE :  847671.25
# R2 :  0.9885200561847298
# C_1월 18일, 19일 :  [[89026.43 89231.51]]
# C_1월 19일 삼성전자 시가 :  [89231.51]

# loss :  915256.875
# mae :  685.6292724609375
# RMSE :  915256.8
# R2 :  0.9876060671604254
# C_1월 18일, 19일 :  [[88751.09 88254.2 ]]
# C_1월 19일 삼성전자 시가 :  [88254.2]

# loss :  955883.9375
# mae :  703.3825073242188
# RMSE :  955884.2
# R2 :  0.9870399955413354
# C_1월 18일, 19일 :  [[88359.336 88455.2  ]]
# C_1월 19일 삼성전자 시가 :  [88455.2]

# loss :  934618.375
# mae :  694.1921997070312
# RMSE :  934622.1
# R2 :  0.9873253002075499
# C_1월 18일, 19일 :  [[84339.83 84748.36]]
# C_1월 19일 삼성전자 시가 :  [84748.36]
