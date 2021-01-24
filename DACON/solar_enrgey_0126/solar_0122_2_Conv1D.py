# 상관계수 높은 열만 사용하기
# 7일치로 2일 예측하기

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

##############################################################

# 만들고 싶은 모양 : 7일치 데이터로 2일치를 예측한다.
# print(x.shape)     # (N, 336, 8)
# print(y.shape)     # (N, 48, 2)
# print(x_pred.shape)  # (81, 336, 8)

##############################################################

# 파일 불러오기
train = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.shape)  # (52560, 9)
# print("df_train null : ", train.duplicated().sum())   # 0

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
    temp = temp[['TARGET','GHI','DHI','DNI','RH','T']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96, :]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train==False:     
        return temp.iloc[:, :]  # 7일 전부 다


# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps, y_row) :  # 336, 48
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end = i + time_steps
        y_end = i + y_row
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i : x_end, :-2]   # ['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T']
        tmp_y = dataset[i : y_end, -2:]   # ['Target1', 'Target2']
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

##############################################################
# train data
df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 8)
# print(df_train.columns) 
# Index(['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T', 'Target1', 'Target2'], dtype='object')


# 상관계수 확인
# print(df_train.corr())
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
# sns.heatmap(data=df_train.corr(),square=True, annot=True, cbar=True)
# plt.show()
# > 기준 : Target1, Target2
# > 상관계수 0.5 이상 : Target, GHI, DHI, DNI, RH, T


X = df_train.to_numpy()
# print(X.shape)      # (52464, 8)

# x, y 데이터 분리
x, y = split_xy(X, 336, 48)
print("x.shape : ", x.shape)     # (52129, 336, 6) : 336행(7일씩), 6열
# print(x[15:18])
# print(x)

print("y.shape : ", y.shape)       # (52129, 48, 2)   : 하루를 한 칼람 > 한꺼번에 이틀
# print(y[15:18])  

# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # print(temp.columns) # Index(['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T'], dtype='object')
    df_test.append(temp)

df_test = pd.concat(df_test)
# print(df_test.shape) # (27216, 6)
# print("df_test null : ", df_test.duplicated().sum())    # 778

x_pred = df_test.to_numpy()

x_pred = x_pred.reshape(81, 336, 6)
print("x_pred.shape : " , x_pred.shape)  # (81, 336, 6)
# print(x_pred)
# print(x_pred[15:18])

##############################################################
# x >> preprocessing
from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = \
#     train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val, = \
    train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    # (33362, 336, 6)
# print(x_test.shape)     # (10426, 336, 6)
# print(x_val.shape)      # (8341, 336, 6)

# print(y_train.shape)   # (33362, 48, 2)
# print(y_test.shape)    # (10426, 48, 2)
# print(y_val.shape)     # (8341, 48, 2)

# StandardScaler를 하기 위해서 2차원으로 변환
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1] * x_pred.shape[2])

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], 336, 6)
# x_test = x_test.reshape(x_test.shape[0], 336, 6)
x_val = x_val.reshape(x_val.shape[0], 336, 6)
x_pred = x_pred.reshape(x_pred.shape[0], 336, 6)


##############################################################

#2. Modeling
#3. Compile, Train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#2. Modeling
def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 336, 6)
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((48,2)))  # output (N, 48, 2)
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2))
    return model

##############################################################

# loss_list = list()

for q in quantiles :
    print("\n>>>>>>>>>>>>>>>>>>>>>> modeling start (q_%.1f)  >>>>>>>>>>>>>>>>>>>>>>" %q) 

    #2. Modeling
    model = modeling()
    model.summary()

    cp_save = '../data/modelcheckpoint/solar_0122_q_{val_loss:.4f}.hdf5'
    es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
    cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.3, verbose=1)

    #3. Compile, Train
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val), callbacks=[es, cp, lr])

    # 4. Evaluate, Predict
    # loss = model.evaluate(x_test, y_test,batch_size=64)
    # print("(q_%.1f) loss : %.4f" % (q, loss))
    # loss_list.append(loss)  # loss 기록

    y_pred = model.predict(x_pred)
    # print(y_pred.shape) # (81, 48, 2)
    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1] , y_pred.shape[2]) # (3888, 2)
    # print(y_pred.shape) #(3888, 2)

    print(y_pred[:100,0])
    print(y_pred[100:200,0])
    print(y_pred[200:300,0])
    
    # submission
    column_name = 'q_' + str(q)
    submission.loc[submission.id.str.contains("Day7"), column_name] = np.around(y_pred[:, 0],3)   # Day7 (3888, 9)
    submission.loc[submission.id.str.contains("Day8"), column_name] = np.around(y_pred[:, 1],3)   # Day8 (3888, 9)
    submission.to_csv('../data/DACON_0126/submission_0122_%f.csv', index=False %q)  # score : 	


# loss_mean = sum(loss_list) / len(loss_list) # 9개 loss 평균
# print(loss_mean)    # 2.0906564791997275

"""
# to csv
submission.to_csv('../data/DACON_0126/submission_0122_1.csv', index=False)  # score : 	
"""