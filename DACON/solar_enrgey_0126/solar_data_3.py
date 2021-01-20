# 3차원 Conv1D

import numpy as np
import pandas as pd

################################

# 예측할 Target 칼럼 추가하기
# GHI = DHI + DNI
def preprocess_data (data, is_train=True) :
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train == True :    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날 TARGET을 붙인다.
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날 TARGET을 붙인다.
        temp.insert(2,'GHI',data['DNI']+data['DHI'])
        temp = temp.drop(['DHI', 'DNI'],axis=1)
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 이틀치 데이터만 빼고 전체
    elif is_train == False :         
        # Day, Minute 컬럼 제거
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        temp.insert(2,'GHI',data['DNI']+data['DHI'])
        temp = temp.drop(['DHI', 'DNI'],axis=1)
        return temp.iloc[-48:, :] # 마지막 하루치 데이터

# 시계열 데이터로 자르기
def split_xy(dataset, time_steps, y_row) :
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_row
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :-2]
        tmp_y = dataset[i:x_end_number, -2:]
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
# Index(['Hour', 'TARGET', 'GHI', 'WS', 'RH', 'T', 'Target1', 'Target2'], dtype='object')
# print(df_train.shape)      # (52464, 8)

dataset = df_train.to_numpy()
# print(dataset.shape)      # (52464, 8)
# print(dataset[0])
# [  0.     0.     0.     1.5   69.08 -12.     0.     0.  ]

x = dataset.reshape(-1, 48, 8)  # 하루치로 나눔
# print(x.shape)  # (1093, 48, 8)
# print(x[0])     # day0

x, y = split_xy(dataset, 48 , 1)
# print(x.shape)     # (52416, 48, 6)  # day0 ~ day7, 7일씩 자름
# print(x[0:3])

# print(y.shape)     # (52416, 48, 2)
# print(y[0:2])  

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
# print(df_pred.shape) # (3888, 6) -> 23328
# print(df_pred.head())

pred_dataset = df_pred.to_numpy()

x_pred = pred_dataset.reshape(81, 48, 6)
# print(x_pred.shape) # (81, 48, 6)

################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    # (33545, 48, 6)
# print(x_test.shape)     # (10484, 48, 6)
# print(x_val.shape)      # (8387, 48, 6)

# print(y_train.shape)    # (33545, 48, 2)
# print(y_test.shape)     # (10484, 48, 2)
# print(y_val.shape)      # (8387, 48, 2)

x_train = x_train.reshape(x_train.shape[0]*x_train.shape[1], x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0]*x_test.shape[1], x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0]*x_val.shape[1], x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0]*x_pred.shape[1], x_pred.shape[2])

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(33545, 48, 6)
x_test = x_test.reshape(10484, 48, 6)
x_val = x_val.reshape(8387, 48, 6)
x_pred = x_pred.reshape(81, 48, 6)

y_train = y_train.reshape(33545, 48, 2)
y_test = y_test.reshape(10484, 48, 2)
y_val = y_val.reshape(8387, 48, 2)

################################

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

for q in q_lst:
    #2. Modeling
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu',\
        input_shape=(x_train.shape[1], x_train.shape[2])))  #input (N, 48, 7)
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((48, 2)))
    model.add(Dense(2, activation='relu'))
    # model.summary()

    #3. Compile, train                  # y : 실제값, pred : 예측값 ????
    model.compile(loss=lambda y, pred: quantile_loss(q,y,pred), optimizer='adam')
    hist = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_val, y_val), callbacks=[es,cp,lr])

    #4. Evaluate, Predict
    result = model.evaluate(x_test, y_test, batch_size=128)
    print("(q_%.2f) loss : %f" % (q, result))   # (q_0.10) loss : 1.799708 <--- 이런 식으로 프린트

    y_pred = model.predict(x_pred)
    print("y_pred : ", y_pred)
    # print(y_pred.shape) # (81, 48, 2)

    # quatile에 따라 다르게 나오는 결과값 저장
    pred_concat.append(y_pred)  
    

pred_concat = np.array(pred_concat) # list형태를 numpy 변환
# print(pred_concat.shape)    # (9, 81, 48, 2)
pred_concat = pred_concat.reshape(9, 81 * 48, 2)

# submission 파일에 결과값 넣기
for j in range(9):
    column_name = 'q_0.' + str(j+1)
    sub.loc[sub.id.str.contains("Day7"), column_name] = pred_concat[j,:,0]
    sub.loc[sub.id.str.contains("Day8"), column_name] = pred_concat[j,:,1]


sub.to_csv('../data/DACON_0126/submission_0120_1.csv', index=False) # 점수 : 3.593266095