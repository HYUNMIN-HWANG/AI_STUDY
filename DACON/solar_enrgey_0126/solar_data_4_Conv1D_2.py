

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

##############################################################

# 파일 불러오기
train = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.shape)  # (52560, 9)
submission = pd.read_csv('../data/DACON_0126/sample_submission.csv')
# print(submission.shape) # (7776, 10)

##############################################################

#1. DATA

# 함수 : GHI column 추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    return data

# 함수 : train data column 정리
# 끝에 다음날, 다다음날 TARGET 데이터를 column을 추가한다.
def preprocess_data(data, is_train=True):
    data = Add_features(data)
    temp = data.copy()
    temp = temp[['Hour','TARGET','GHI','DHI','DNI','WS','RH','T']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train==False:     
        return temp.iloc[-48:, :]   # 0 ~ 6일 중 마지막 6일 데이터만 남긴다. (6일 데이터로 7, 8일을 예측하고자 함) 

# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps) :
    x, y1, y2 = list(), list(), list()
    for i in range(len(dataset)) :
        x_end = i + time_steps
        # y_end_number = x_end_number + y_row
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i:x_end, :-2]            # ['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']
        tmp_y1 = dataset[x_end-1 : x_end, -2]    # ['Target1']
        tmp_y2 = dataset[x_end-1 : x_end, -1]    # ['Target2'] 
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return np.array(x), np.array(y1), np.array(y2)

##############################################################

df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 10)
# print(df_train.columns) 
# Index(['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1','Target2'], dtype='object')

X = df_train.to_numpy()
# print(X.shape)      # (52464, 10)

# 3차원으로 바꾸기 전에 2차원에서 StandardScaler >> x_train에만 적용하도록 바꾸기 *************
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# x, y 데이터 분리
x, y1, y2 = split_xy(X, 1)
# print(x.shape)     # (52464, 1, 8)  # 한 행씩 자름
# print(x[15:18])

# print(y1.shape)     # (52464, 1)
# print(y1[15:18])  
# print(y2.shape)     # (52464, 1)
# print(y2[15:18])  

# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # print(temp.columns) # Index(['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T'], dtype='object')
    df_test.append(temp)

x_test = pd.concat(df_test)
# print(X_test.shape) # (3888, 8)
x_pred = x_test.to_numpy()

x_pred = x_pred.reshape(3888, 1, 8)
# print(x_pred.shape)  # (3888, 1, 8)
# print(x_pred[15:18])

##############################################################
# x >> preprocessing
from sklearn.model_selection import train_test_split

x_train, x_test, y1_train, y1_test, y2_train, y2_test = \
    train_test_split(x, y1, y2, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y1_train, y1_val, y2_train, y2_val = \
    train_test_split(x_train, y1_train, y2_train, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    # (33576, 1, 8)
# print(x_test.shape)     # (10493, 1, 8)
# print(x_val.shape)      # (8395, 1, 8)
# print(y1_train.shape)   # (33576, 1)
# print(y1_test.shape)    # (10493, 1)
# print(y1_val.shape)     # (8395, 1)

##############################################################

#2. Modeling
#3. Compile, Train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor
import tensorflow.keras.backend as K

cp_save = '../data/modelcheckpoint/solar_0121_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=10, mode='min')
cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# 함수 : Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#2. Modeling
def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    return model

#3. Compile, Train
def train_data(x_train, x_test, y_train, y_test, x_val, y_val, x_pred):
    pred_list = []
    loss_list = list()

    for q in quantiles:
        print("(q_%.1f) modeling start : >>>>>>>>>>>>>>>>>>>>>>" %q) 
        #2. Modeling
        model = modeling()

        #3. Compile, Train
        model.compile(loss = lambda y_true,y_pred: quantile_loss(q,y_true,y_pred), \
            optimizer = 'adam', metrics = [lambda y_true,y_pred: quantile_loss(q,y_true,y_pred)])
        model.fit(x_train, y_train, epochs=120, batch_size=64, validation_data=(x_val, y_val), callbacks=[es, cp, lr])

        # 4. Evaluate, Predict
        loss = model.evaluate(x_test, y_test,batch_size=64)
        print("(q_%.1f) loss : %.4f" % (q,loss[0]))
        loss_list.append(loss[0])

        pred = pd.DataFrame(model.predict(x_pred).round(2))
        pred_list.append(pred)
    
    loss_mean = sum(loss_list) / len(loss_list)     # loss의 평균 저장

    df_temp = pd.concat(pred_list, axis = 1)        # 예측값인 result 저장
    df_temp[df_temp<0] = 0
    result = df_temp.to_numpy()
    return result, loss_mean

##############################################################

# Target1 결과값 저장
results_1, loss_mean1 = train_data(x_train, x_test, y1_train, y1_test, x_val, y1_val, x_pred)
# Target2 결과값 저장
results_2, loss_mean2 = train_data(x_train, x_test, y2_train, y2_test, x_val, y2_val, x_pred)

# print(results_1.shape, results_2.shape) # (3888, 9) (3888, 9)

print(loss_mean1)
print(loss_mean2)

# 0.07797420128352112
# 0.07946157910757595

##############################################################

# submission 저장
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1    # Day7 (3888, 9)
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2    # Day8 (3888, 9)

submission.to_csv('../data/DACON_0126/submission_0121_1.csv', index=False)  # score : 7.8279809707 왜!!!!!!!!!!!!??/ 전처리 잘못함
