# 4차원 Conv2D

import numpy as np
import pandas as pd

# quantile 안 됨 xxxxxxxxxxxxxxxxxxxxxxxx

# 목표
# x : (1089, 7, 48, 6)  # 7일치 데이터
# y : (1089, 2, 48, 1)  # 1일치 데이터
# x_pred : (81, 7, 48, 6)
# y_pred : (81, 2, 48, 1)

################################

# Day, Minute 컬럼 제거
# GHI = DHI + DNI 칼럼 추가하기
def preprocess_data (data, is_train=True) :
    temp = data.copy()
    temp = temp[['Hour','DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET']]

    if is_train == True :    
        temp.insert(2,'GHI',data['DNI']+data['DHI'])
        temp = temp.drop(['DHI', 'DNI'],axis=1)
        # temp = temp.dropna()    # 결측값 제거 <- 문제. 제거하면 행 수가 달라져서 처리하기 곤란 (제거하는 것보다 비워있는 곳을 채우는 건 어떨까??)
        temp = temp.fillna(method='ffill') # 결측치를 앞에 있는 숫자로 채워준다.
        return temp

    elif is_train == False :         
        temp.insert(2,'GHI',data['DNI']+data['DHI'])
        temp = temp.drop(['DHI', 'DNI'],axis=1)
        temp = temp.fillna(method='ffill') # 결측치를 앞에 있는 숫자로 채워준다.
        return temp

# 시계열 데이터로 자르기
def split_xy(data, time_steps, y_col):
    x,y=list(), list()
    for i in range(len(data)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_col
        if y_end_number > len(data):
            break
        tmp_x=data[i:x_end_number, :]
        tmp_y=data[x_end_number-1:y_end_number-1, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

################################

#1. DATA

# train 데이터 불러오기 >> x_train
train_pd = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train_pd.columns)    # Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# print(train_pd.shape)      # (52560, 9)
df_train = preprocess_data(train_pd)
# print(df_train.columns) 
# Index(['Hour', 'GHI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# print(df_train.shape)      # (52560, 6)

dataset = df_train.to_numpy()
# print(dataset.shape)      # (52560, 6)
# print(dataset[0:3])
# [[  0.     0.     1.5   69.08 -12.     0.  ]
#  [  0.     0.     1.5   69.06 -12.     0.  ]
#  [  1.     0.     1.6   71.78 -12.     0.  ]]

x = dataset.reshape(-1, 48, 6)  # 하루치로 나눔
# print(x.shape)  # (1095, 48, 6)
# print(x[0])     # day0

x, y = split_xy(dataset, 336 , 96)

x = x.reshape(x.shape[0], 7, 48, 6)
# print(x.shape)     # (52129, 7, 48, 6)  # day0 ~ day7, 7일씩 자름
# print(x[2])
y = y.reshape(y.shape[0], 2, 48,1)
# print(y.shape)     # (52129, 2, 48, 1)
# print(y[1])  

# submission file 불러오기
sub = pd.read_csv('../data/DACON_0126/sample_submission.csv')

################################

# test 데이터 불러오기 >> x_pred
df_pred = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_pred.append(temp)

df_pred = pd.concat(df_pred)
# print(df_pred.shape) # (27216, 6)
# print(df_pred.head())

pred_dataset = df_pred.to_numpy()

x_pred = pred_dataset.reshape(81, 7, 48, 6)
# print(x_pred.shape) # (81, 7, 48, 6)

################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66)

print(x_train.shape)    # (33362, 7, 48, 6)
print(x_test.shape)     # (10426, 7, 48, 6)
print(x_val.shape)      # (8341, 7, 48, 6)

print(y_train.shape)    # (33362, 2, 48, 1)
print(y_test.shape)     # (10426, 2, 48, 1)
print(y_val.shape)      # (8341, 2, 48, 1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2] * x_val.shape[3])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1] * x_pred.shape[2] * x_pred.shape[3])

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(33362, 7, 48, 6)
x_test = x_test.reshape(10426, 7, 48, 6)
x_val = x_val.reshape(8341, 7, 48, 6)
x_pred = x_pred.reshape(81, 7, 48, 6)

y_train = y_train.reshape(33362, 2, 48, 1)
y_test = y_test.reshape(10426, 2, 48, 1)
y_val = y_val.reshape(8341, 2, 48, 1)

################################

#2. Modeling

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPool1D, MaxPool2D, Dropout, Reshape
from tensorflow.keras.backend import mean, maximum
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

modelpath = '../data/modelcheckpoint/solar_0120_{epoch:02d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', patience=20, mode='min')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True ,mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

# quantile_loss 
def quantile_loss(q, y, pred):
    err = (y-pred)
    return mean(maximum(q*err, (q-1)*err), axis=-1)

q_lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pred_concat = list()

# quantile 0.1 ~ 0.9 총 9번 반복
for q in q_lst:
    #2. Modeling
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu',\
        input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3])))  #input (N, 7, 48, 6)
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((2, 48, 1)))
    model.add(Dense(1, activation='relu'))
    # model.summary()

    #3. Compile, train     
    model.compile(loss=lambda y, pred: quantile_loss(q,y,pred), optimizer='adam')
    hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), callbacks=[es,cp,lr])

    #4. Evaluate, Predict
    result = model.evaluate(x_test, y_test, batch_size=128)
    print("(q_%.1f) loss : %f" % (q, result))   # (q_0.1) loss : 1.799708 <--- 이런 식으로 프린트

    y_pred = model.predict(x_pred)
    # print("y_pred : ", y_pred)
    # print(y_pred.shape) # (81, 2, 48, 1)
    y_pred = y_pred.reshape(7776,1)

    # # quatile에 따라 다르게 나오는 결과값 저장
    column_name = 'q_' + str(q)
    print(column_name)
    sub.loc[:, column_name] = y_pred 

# to csv
sub.to_csv('../data/DACON_0126/submission_0120_1.csv', index=False) # score : 	2.0202123047	
