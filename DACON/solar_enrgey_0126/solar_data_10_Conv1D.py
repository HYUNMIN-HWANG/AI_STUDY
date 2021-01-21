import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

##############################################################

# print(x.shape)     # (N, 336, 8)
# print(y.shape)     # (N, 48, 2)
# print(x_pred.shape)  # (81, 336, 8)

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
        return temp.iloc[:, :]


# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps, y_row) :
    x, y1 = list(), list()
    for i in range(len(dataset)) :
        x_end = i + time_steps
        y_end = x_end
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i:x_end, :-2]                # ['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T']
        tmp_y1 = dataset[x_end-y_row : y_end, -2:]    # ['Target1', 'Target2']
        x.append(tmp_x)
        y1.append(tmp_y1)
    return np.array(x), np.array(y1)

##############################################################

df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 10)
# print(df_train.columns) 
# Index(['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1','Target2'], dtype='object')

X = df_train.to_numpy()
# print(X.shape)      # (52464, 10)

# x, y 데이터 분리
x, y = split_xy(X, 336, 48)
print(x.shape)     # (52129, 336, 8) : 7일씩
# print(x[15:18])

print(y.shape)     #(52129, 48, 2)   : 하루를 한 칼람 > 한꺼번에 이틀
# print(y[15:18])  


# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # print(temp.columns) # Index(['Hour', 'TARGET', 'GHI', 'DHI', 'DNI', 'WS', 'RH', 'T'], dtype='object')
    df_test.append(temp)

df_test = pd.concat(df_test)
# print(X_test.shape) # (3888, 8)
x_pred = df_test.to_numpy()

x_pred = x_pred.reshape(81, 336, 8)
# print(x_pred.shape)  # (81, 336, 8)
# print(x_pred[15:18])

##############################################################
# x >> preprocessing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)
x_train, x_val, y_train, y_val, = \
    train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=66)

# print(x_train.shape)    # (33362, 336, 8)
# print(x_test.shape)     # (10426, 336, 8)
# print(x_val.shape)      # (8341, 336, 8)

# print(y_train.shape)   # (33362, 48, 2)
# print(y_test.shape)    # (10426, 48, 2)
# print(y_val.shape)     # (8341, 48, 2)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1] * x_pred.shape[2])

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(x_train.shape[0], 336, 8)
x_test = x_test.reshape(x_test.shape[0], 336, 8)
x_val = x_val.reshape(x_val.shape[0], 336, 8)
x_pred = x_pred.reshape(x_pred.shape[0], 336, 8)


##############################################################

#2. Modeling
#3. Compile, Train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, MaxPool1D,Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from lightgbm import LGBMRegressor
import tensorflow.keras.backend as K

cp_save = '../data/modelcheckpoint/solar_0121_{val_loss:.4f}.hdf5'
es = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, verbose=1)

# 함수 : Quantile loss definition
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#2. Modeling
def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(96, activation='relu'))
    model.add(Reshape((48,2)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2))
    return model

#3. Compile, Train

loss_list = list()

for q in quantiles :
    print("(q_%.1f) modeling start : >>>>>>>>>>>>>>>>>>>>>>" %q) 

    #2. Modeling
    model = modeling()
    model.summary()
    #3. Compile, Train
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')
    model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val), callbacks=[es, cp, lr])

    # 4. Evaluate, Predict
    loss = model.evaluate(x_test, y_test,batch_size=128)
    print("(q_%.1f) loss : %.4f" % (q, loss))
    loss_list.append(loss)

    y_pred = model.predict(x_pred)
    # print(y_pred.shape) # (81, 48, 2)
    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1] , y_pred.shape[2]) # (3888, 2)
    print(y_pred.shape) #
    # print(y_pred[:,0])
    
    # submission
    column_name = 'q_' + str(q)
    submission.loc[submission.id.str.contains("Day7"), column_name] = y_pred[:, 0].round(4)    # Day7 (3888, 9)
    submission.loc[submission.id.str.contains("Day8"), column_name] = y_pred[:, 1].round(4)    # Day8 (3888, 9)

loss_mean = sum(loss_list) / len(loss_list)
print(loss_mean)    # 2.008312543233236
submission.to_csv('../data/DACON_0126/submission_0121_3.csv', index=False)  # score : 	

"""
def train_data(x_train, x_test, y_train, y_test, x_val, y_val, x_pred):
    pred_list = []
    loss_list = list()

    for q in quantiles:
        print("(q_%.1f) modeling start : >>>>>>>>>>>>>>>>>>>>>>" %q) 
        #2. Modeling
        model = modeling()
        model.summary()
        #3. Compile, Train
        model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')
        model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_data=(x_val, y_val), callbacks=[es, cp, lr])

        # 4. Evaluate, Predict
        loss = model.evaluate(x_test, y_test,batch_size=64)
        print("(q_%.1f) loss : %.4f" % (q, loss))
        loss_list.append(loss)

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

# 1.9882435997327168
# 2.030441893471612

##############################################################

# submission 저장
submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1    # Day7 (3888, 9)
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2    # Day8 (3888, 9)

submission.to_csv('../data/DACON_0126/submission_0121_3.csv', index=False)  # score : 1.9855	
"""