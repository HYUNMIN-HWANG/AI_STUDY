
# 9 quantile * target2개 * 100 epoch >>> 시간이 너무 오래 걸린다.
# 결과값이 왜 음수가 나오는 거지????????????????

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

##############################################################

# 파일 불러오기
train = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.shape)  # (52560, 9)
submission = pd.read_csv('../data/DACON_0126/sample_submission.csv')
# print(submission.shape) # (7776, 10)

##############################################################

#1. DATA

# 함수 : train data column 정리
# 끝에 다음날, 다다음날 TARGET 데이터를 column을 추가한다.
def preprocess_data(data, is_train=True):
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)
    elif is_train==False:
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]       
        return temp.iloc[-48:, :]   # 0 ~ 6일 중 마지막 6일 데이터만 남긴다. (6일 데이터로 7, 8일을 예측하고자 함) 

# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps, y_row) :
    x, y1, y2 = list(), list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_row
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-2]
        tmp_y1 = dataset[i:x_end_number, -2]    # ['Target1']
        tmp_y2 = dataset[i:x_end_number, -1]    # ['Target2'] 
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

##############################################################

df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 9)
# print(df_train.columns) 
# Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1', 'Target2'], dtype='object')

dataset = df_train.to_numpy()
# print(dataset.shape)      # (52464, 9)
x = dataset.reshape(-1, 48, 9)  # 하루치로 나눔

x, y1, y2 = split_xy(dataset, 48 , 1)
# print(x.shape)     # (52416, 48, 7)  # day0 ~ day7, 7일씩 자름
# print(x[0:3])

# print(y1.shape)     # (52416, 48)
# print(y1[0:2])  
# print(y2.shape)     # (52416, 48)
# print(y2[0:2])  

# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
# print(X_test.shape) # (3888, 7)
X_pred = X_test.to_numpy()
X_pred = X_pred.reshape(81, 48, 7)
# print(X_pred.shape)   # (81, 48, 7)

##############################################################
# x >> preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(x, y1, test_size=0.3, shuffle=True, random_state=66)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(x, y2, test_size=0.3, shuffle=True, random_state=66)

# print(X_train_1.shape)  # (36691, 48, 7)
# print(X_valid_1.shape)  # (15725, 48, 7)

# print(Y_train_1.shape)  # (36691, 48)  
# print(Y_valid_1.shape)  # (15725, 48) 

X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1] * X_train_1.shape[2])
X_valid_1 = X_valid_1.reshape(X_valid_1.shape[0], X_valid_1.shape[1] * X_valid_1.shape[2])
X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1] * X_train_2.shape[2])
X_valid_2 = X_valid_2.reshape(X_valid_2.shape[0], X_valid_2.shape[1] * X_valid_2.shape[2])
X_pred    = X_pred.reshape   (X_pred.shape[0],    X_pred.shape[1]*X_pred.shape[2])

scaler = MinMaxScaler()
scaler.fit(X_train_1)
scaler.fit(X_train_2)
X_train_1 = scaler.transform(X_train_1)
X_valid_1 = scaler.transform(X_valid_1)
X_train_2 = scaler.transform(X_train_2)
X_valid_2 = scaler.transform(X_valid_2)
X_pred = scaler.transform(X_pred)

X_train_1 = X_train_1.reshape(36691, 48, 7)
X_valid_1 = X_valid_1.reshape(15725, 48, 7)
X_train_2 = X_train_2.reshape(36691, 48, 7)
X_valid_2 = X_valid_2.reshape(15725, 48, 7)
X_pred = X_pred.reshape(81, 48, 7)

# y_train = y_train.reshape(33545, 48, 2)
# y_test = y_test.reshape(10484, 48, 2)
# y_val = y_val.reshape(8387, 48, 2)


##############################################################

#2. Modeling
#3. Compile, Train
################################## 모델링 보강#################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor
from tensorflow.keras.backend import mean, maximum

es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# 함수 : Quantile loss definition
def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 함수 : Modeling, Compile, Train
def conv(q, X_train, Y_train, X_valid, Y_valid, X_pred):
    # (a) Modeling  
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',\
         input_shape=(X_train_1.shape[1], X_train_1.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(48))

    model.compile(loss = lambda y, pred: quantile_loss(q, y, pred), \
        optimizer='adam', metrics=['mae'] )
    model.fit(X_train, Y_train, epochs=100, batch_size=128,\
            validation_data =(X_valid, Y_valid), callbacks=[es,lr], verbose=1)
    
    # (b) Predictions
    pred = pd.DataFrame(model.predict(X_pred).round(2)) # <-- 확인 : DataFrame으로 안 받아도 될 거 같은데...?
    # print("pred.shape : " ,pred.shape)  # (81, 48)
    pred = pred.to_numpy()
    # print("pred.shape : " ,pred.shape)  # (81, 48)
    pred = pred.reshape(3888,)

    return pred, model

# 함수 : Target 예측 (quantiles 9개 -> 9번 모델링 반복)
def train_data(X_train, Y_train, X_valid, Y_valid, X_pred):
    conv_models=[]
    conv_actual_pred = list()

    for q in quantiles:
        print(q)
        pred , model = conv(q, X_train, Y_train, X_valid, Y_valid, X_pred)
        conv_models.append(model)
        conv_actual_pred.append(pred)
    conv_actual_pred = np.array(conv_actual_pred) # numpy 형태를 list 변환
    # LGBM_actual_pred.columns=quantiles
    return conv_models, conv_actual_pred

##############################################################

# Target1 결과값 저장
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_pred)

# Target2 결과값 저장
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_pred)

# print(results_1.shape, results_2.shape) # (9, 3888) (9, 3888)

results_1 = np.transpose(results_1) # 행, 열 바꾸기
results_2 = np.transpose(results_2)
# print(results_1.shape, results_2.shape) # (3888, 9) (3888, 9)

##############################################################

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1    # Day7 (3888, 9)
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2    # Day8 (3888, 9)


submission.to_csv('../data/DACON_0126/submission_0120_2.csv', index=False)  # score : 2.7490271989
