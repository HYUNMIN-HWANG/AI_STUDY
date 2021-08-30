# 상관계수 높은 열만 사용하기
# 하루치 데이터로 이틀치를 예측

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

##############################################################

# 만들고 싶은 모양 : 하루치 데이터로 이틀치를 예측한다.
# print(x.shape)     # (N, 48, 6)
# print(y.shape)     # (N, 48, 2)
# print(x_pred.shape)  # (81, 48, 6)

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
        # print(temp.isnull().sum()) # 결측값 없음
        temp = temp.dropna()       # 결측값 제거
        return temp.iloc[:-96, :]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train==False:     
        # print(temp.isnull().sum()) # 결측값 없음
        return temp.iloc[-48:, :]  # 하루 전부 다

# 5. 결측값이 들어있는 행 전체 제거
# print(df_sorted.isnull().sum())    
# null : 2018-04-30, 2018-05-02, 2018-05-03 >> 거래량  3개 / 금액(백만) 3개
# df_drop_null = df_sorted.dropna(axis=0)
# print(df_drop_null.shape)    # (2399, 15)
# print(df_drop_null.isnull().sum())  # null 제거 확인


# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, x_row, x_col, y_row, y_col) :  # 48, 48
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end = i + x_row
        y_end = x_end
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i:x_end, :x_col]   # ['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T']
        tmp_y = dataset[y_end - y_row : y_end, -y_col:]  # ['Target1', 'Target2']
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

##############################################################
# train data
df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 8)
# print(df_train.columns) 
# Index(['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T', 'Target1', 'Target2'], dtype='object')
# print(df_train[:25])

# 상관계수 확인
# print(df_train.corr())
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
# sns.heatmap(data=df_train.corr(),square=True, annot=True, cbar=True)
# plt.show()
# > 기준 : Target1, Target2
# > 상관계수 0.5 이상 : Target, GHI, DHI, DNI, RH, T

X = df_train.values
# X = df_train.to_numpy()
# print(X.shape)      # (52464, 8)
# print(X[:25])

# x, y 데이터 분리
x, y = split_xy(X, 2, 6, 2, 2)
print("x.shape : ", x.shape)     # (52463, 2, 6) 
# print(x[22:25])

print("y.shape : ", y.shape)       #  (52463, 2, 2)
# print(y[22:25])  

# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # print(temp.columns) # Index(['TARGET', 'GHI', 'DHI', 'DNI', 'RH', 'T'], dtype='object')
    df_test.append(temp)

df_test = pd.concat(df_test)
# print(df_test.shape) # (3888, 6)
x_pred = df_test.values
# x_pred = df_test.to_numpy()
# print(x_pred[22:25])


x_pred = x_pred.reshape(-1, 2, 6)
print("x_pred.shape : " , x_pred.shape)  # (1944, 2, 6)
# print(x_pred[15:18])

##############################################################
# x >> preprocessing

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])  
y = y.reshape(y.shape[0], y.shape[1] * y.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred .shape[1] * x_pred.shape[2])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, shuffle=True, random_state=113)
x_train, x_val, y_train, y_val, = \
    train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=113)

# StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], 2, 6)
x_test = x_test.reshape(x_test.shape[0], 2, 6)
x_val = x_val.reshape(x_val.shape[0], 2 , 6)
x_pred = x_pred.reshape(x_pred.shape[0], 2, 6)

y_train = y_train.reshape(y_train.shape[0], 2, 2)
y_test = y_test.reshape(y_test.shape[0], 2, 2)
y_val = y_val.reshape(y_val.shape[0], 2, 2)

# print(x_train.shape)    # (33576, 2, 6)
# print(x_test.shape)     # (10493, 2, 6)
# print(x_val.shape)      # (8394, 2, 6)
# print(x_pred.shape)      # (1944, 2, 6)

# print(y_train.shape)   # (33576, 2, 2)
# print(y_test.shape)    # (10493, 2, 2)
# print(y_val.shape)     # (8394, 2, 2)

# print(x_train[:10])


##############################################################

#2. Modeling
#3. Compile, Train
from tensorflow.keras.models import Sequential, load_model
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
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 336, 6)
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((48,2)))  # output (N, 48, 2)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2))
    return model

##############################################################

loss_list = list()

for q in quantiles :
    print(f"\n>>>>>>>>>>>>>>>>>>>>>> modeling start 'q_{q}'  >>>>>>>>>>>>>>>>>>>>>>") 

    #2. Modeling
    # model = modeling()
    cp_load = f'../data/DACON_cp/save/solar_0123_1_q{q:.1f}.hdf5'
    model = load_model(cp_load, compile = False)
    model.summary()

    #3. Compile, Train
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam',  metrics=['mse'])
    
    # es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    # lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, verbose=1)
    # cp_save = f'../data/modelcheckpoint/solar_0122_q_{q:.1f}.hdf5'
    # cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, mode='min')
    # hist = model.fit(x_train, y_train, epochs=500, batch_size=64, validation_data=(x_val, y_val), callbacks=[es, cp, lr])

    # 4. Evaluate, Predict
    result = model.evaluate(x_test, y_test,batch_size=64)
    print('loss: ', result[0])
    print('mae: ', result[1])
    loss_list.append(result[0])  # loss 기록

    y_pred = model.predict(x_pred)
    # print(y_pred.shape) # (81, 48, 2)
    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])) # (3888, 2)
    # print(y_pred.shape) #(3888, 2)
    y_pred = pd.concat([y_pred], axis=1)
    y_pred[y_pred<0] = 0
    y_pred = y_pred.to_numpy()

    # print(y_pred[:100,0])
    # print(y_pred[100:200,0])
    # print(y_pred[200:300,0])
    
    # submission
    # column_name = 'q_' + str(q)
    column_name = f'q_{q}'
    submission.loc[submission.id.str.contains("Day7"), column_name] = y_pred[:, 0].round(2)  # Day7 (3888, 9)
    submission.loc[submission.id.str.contains("Day8"), column_name] = y_pred[:, 1].round(2)   # Day8 (3888, 9)


loss_mean = sum(loss_list) / len(loss_list) # 9개 loss 평균
print(loss_mean)   # 0.6755326390266418


# to csv
submission.to_csv('../data/DACON_0126/0123_1.csv', index=False)  # score : 

# 0123_1    
# 0123_2    # 1.9791257977485657
