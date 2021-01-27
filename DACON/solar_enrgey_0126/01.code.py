'''
[Dacon] AI프렌즈 시즌3 SMP 및 전력수요 예측 경진대회
팀명 : xian
제출날짜 : 2020년 1월 일
'''

'''
1. 라이브러리 및 데이터
Library & Data
'''

# Conv1D를 사용해 태양광 발전량을 예측한다.

import pandas as pd
import numpy as np
import os
import glob
import random
import warnings
import tensorflow.keras.backend as K
warnings.filterwarnings("ignore")

##############################################################

# train 파일 불러오기
train = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.shape)  # (52560, 9)

# submission 파일 불러오기
submission = pd.read_csv('../data/DACON_0126/sample_submission.csv')
# print(submission.shape) # (7776, 10)

##############################################################

#1. DATA

# 함수 : GHI, T-Td column 추가
# B조의 한수님이 이거 설명하실 수 있다고 들어서 설명 부탁드려볼 예정
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)   
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis= 1, inplace = True)
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    return data

# 함수 : train/test data column 정리
def preprocess_data(data, is_train=True):
    data = Add_features(data)
    # print(data.columns) 
    # Index(['Day', 'T-Td', 'Td', 'GHI', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH','T', 'TARGET'], dtype='object')
    temp = data.copy()
    temp = temp[['Day','TARGET','GHI','DHI','DNI','T-Td']]                   # Target과 상관계수가 높은 칼럼만 사용한다.

    if is_train==True:                                                       # train data 경우      
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()                                                 # 결측값 제거
        return temp.iloc[:-96, :]                                            # 예측하고자 하는 마지막 이틀 치의 데이터를 제외한다.

    elif is_train==False:                                                    # test data 경우
        return temp.iloc[-48*6:, 1:]                                         # 각 test data의 6일 치만 사용한다.

#함수 : 같은 시간대끼리 모으기
def same_train(train) :
    temp = train.copy()
    x = list()                                      # 각 시간별로 모은 데이터
    final_x = list()                                # 반환하고자 하는 최종 결과
    for i in range(48) :                            # 48개의 시간대 만큼 반복
        same_time = pd.DataFrame()                  # 같은 시간대에 있는 데이터를 저장하기 위한 데이터프레임
        for j in range(int(len(temp)/48)) :         # train data에 있는 모든 일수(1095일)만큼 반복
            tmp = temp.iloc[i + 48*j, : ]           # 같은 시간대에 있는 행만 모으기
            tmp = tmp.to_numpy()
            tmp = tmp.reshape(1, tmp.shape[0])      # 행/열 바꾸기
            tmp = pd.DataFrame(tmp)
            same_time = pd.concat([same_time, tmp]) # 이전 시간대와 합치기
        x = same_time.to_numpy()    
        final_x.append(x)                           # 최종 리스트로 합치기
    return np.array(final_x)                        # numpy로 반환

# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps) :           # time_steps = 6 : 6일 간격으로 잘라서 연속적인 데이터를 만들고자 한다.
    x, y = list(), list()
    for i in range(len(dataset)) :            # dataset 길이만큼 반복
        x_end = i + time_steps                # x : 6일치씩 데이터를 자른다.
        y_end = x_end-1                       # y : 1일치씩 데이터를 자른다.
        if x_end > len(dataset) :             # 데이터 길이를 넘어간다면 for문 종료
            break
        tmp_x = dataset[i : x_end, 1:-2]      # x : ['TARGET', 'GHI', 'DHI', 'DNI', 'T-Td'] 컬럼만 사용
        tmp_y = dataset[y_end, -2:]           # y : ['Target1', 'Target2'] 컬럼만 사용
        x.append(tmp_x)                       # 6일 씩 자른 x 데이터를 합친다.
        y.append(tmp_y)                       # 1일 씩 자른 y 데이터를 합친다.
    return np.array(x), np.array(y)           # x와 y를 각자 numpy로 반환

##############################################################

'''
2. 데이터 전처리
Data Cleansing & Pre-Processing
'''
# 2번과 3번 차이를 모르겠음
'''
3. 탐색적 자료분석
Exploratory Data Analysis
'''

# train data 전처리
df_train = preprocess_data(train)
# print(df_train.shape)     # (52464, 8)
# print(df_train.head())
'''
   Day  TARGET  GHI  DHI  DNI      T-Td  Target1  Target2
0    0     0.0  0.0    0    0  4.522271      0.0      0.0
1    0     0.0  0.0    0    0  4.525742      0.0      0.0
2    0     0.0  0.0    0    0  4.061776      0.0      0.0
3    0     0.0  0.0    0    0  4.066807      0.0      0.0
4    0     0.0  0.0    0    0  3.500215      0.0      0.0
'''

# 상관계수 확인
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
# sns.heatmap(data=df_train.corr(),square=True, annot=True, cbar=True)
# plt.show()
# 기준 : Target1, Target2
# 상관계수 0.6 이상인 컬럼만 사용할 예정 : Target, GHI, DHI, DNI, T-Td

# 같은 시간대 별로 묶기
same_time = same_train(df_train)
# print(same_time.shape)  # (48, 1093, 8)
# print(same_time[0,:10,:])
'''
0시 0분일 때의 Day 0일부터 1092일까지 데이터 확인
[[0.         0.         0.         0.         0.         4.5222713      0.         0.        ]
 [1.         0.         0.         0.         0.         1.23733211     0.         0.        ]
 [2.         0.         0.         0.         0.         3.85633304     0.         0.        ]
 [3.         0.         0.         0.         0.         3.04163913     0.         0.        ]
 [4.         0.         0.         0.         0.         4.25246931     0.         0.        ]
 [5.         0.         0.         0.         0.         3.51091823     0.         0.        ]
 [6.         0.         0.         0.         0.         1.95079117     0.         0.        ]
 [7.         0.         0.         0.         0.         7.88527903     0.         0.        ]
 [8.         0.         0.         0.         0.         2.13855559     0.         0.        ]
 [9.         0.         0.         0.         0.         3.43598851     0.         0.        ]]
'''
# print(same_time[0,-10:,:])
'''
[[1083.      0.         0.         0.         0.         8.5502661      0.         0.        ]
 [1084.      0.         0.         0.         0.         7.46005937     0.         0.        ]
 [1085.      0.         0.         0.         0.         4.78875474     0.         0.        ]
 [1086.      0.         0.         0.         0.         7.06061247     0.         0.        ]
 [1087.      0.         0.         0.         0.         5.26179355     0.         0.        ]
 [1088.      0.         0.         0.         0.         6.46620945     0.         0.        ]
 [1089.      0.         0.         0.         0.         3.17928125     0.         0.        ]
 [1090.      0.         0.         0.         0.         2.78822817     0.         0.        ]
 [1091.      0.         0.         0.         0.         8.48614485     0.         0.        ]
 [1092.      0.         0.         0.         0.         9.00927341     0.         0.        ]]
'''

x, y = list(), list()
for i in range(48):                         # 같은 시간대 별로 시계열 데이터를 만들기 위해 48번 반복
    tmp1,tmp2 = split_xy(same_time[i], 6)   # 같은 시간대 별로 묶인 데이터를 6일치씩 자른다. 
    x.append(tmp1)                          # x끼리 묶기
    y.append(tmp2)                          # y끼리 묶기


x = np.array(x)                             # numpy로 반환
y = np.array(y)                             # numpy로 반환

print(x[0,:5,:,:])
'''
0시 0분일 때의 Day 0일부터 1092일까지 데이터를 시계열로 잘 잘려있는지 확인
[[[0.         0.         0.         0.         4.5222713 ]
  [0.         0.         0.         0.         1.23733211]
  [0.         0.         0.         0.         3.85633304]
  [0.         0.         0.         0.         3.04163913]
  [0.         0.         0.         0.         4.25246931]
  [0.         0.         0.         0.         3.51091823]]

 [[0.         0.         0.         0.         1.23733211]
  [0.         0.         0.         0.         3.85633304]
  [0.         0.         0.         0.         3.04163913]
  [0.         0.         0.         0.         4.25246931]
  [0.         0.         0.         0.         3.51091823]
  [0.         0.         0.         0.         1.95079117]]

 [[0.         0.         0.         0.         3.85633304]
  [0.         0.         0.         0.         3.04163913]
  [0.         0.         0.         0.         4.25246931]
  [0.         0.         0.         0.         3.51091823]
  [0.         0.         0.         0.         1.95079117]
  [0.         0.         0.         0.         7.88527903]]

 [[0.         0.         0.         0.         3.04163913]
  [0.         0.         0.         0.         4.25246931]
  [0.         0.         0.         0.         3.51091823]
  [0.         0.         0.         0.         1.95079117]
  [0.         0.         0.         0.         7.88527903]
  [0.         0.         0.         0.         2.13855559]]

 [[0.         0.         0.         0.         4.25246931]
  [0.         0.         0.         0.         3.51091823]
  [0.         0.         0.         0.         1.95079117]
  [0.         0.         0.         0.         7.88527903]
  [0.         0.         0.         0.         2.13855559]
  [0.         0.         0.         0.         3.43598851]]]
'''

# print("x.shape : ", x.shape) # (48, 1088, 6, 5)
# print("y.shape : ", y.shape) # (48, 1088, 2)
"""
y = y.reshape(48, 1088, 1, 2)               # x와 동일한 shape를 맞추기 위해 y도 4차원으로 reshape


# test data 전처리 
df_test = []                                                
for i in range(81):                                                                         # 81개의 0 ~ 7 Day 데이터 합치기
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'                                # 파일 하나씩 불러오기
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)                                            # 전처리 함수를 사용해 test 데이터 전처리
    # print(temp.columns) # Index(['TARGET', 'GHI', 'DHI', 'DNI', 'T-Td'], dtype='object')  # 최종적으로 사용할 컬럼
    temp = pd.DataFrame(temp)
    temp = same_train(temp)                                                                 # 같은 시간대끼리 묶기
    df_test.append(temp)                                                                    # test 데이터 합치기

x_pred = np.array(df_test)                                # 최종적으로 예측하고자 하는 데이터                                              
# print("x_pred.shape : ", x_pred.shape) # (81, 48, 6, 5)


##############################################################

# train data를 train, test, validation 데이터로 분리한다. 
from sklearn.model_selection import train_test_split                         

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, shuffle=True, random_state=332)  
x_train, x_val, y_train, y_val, = \
    train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=332)

# print(x_train.shape)    # (30, 1088, 6, 5)
# print(x_test.shape)     # (10, 1088, 6, 5)
# print(x_val.shape)      # (8, 1088, 6, 5)

# print(y_train.shape)   # (30, 1088, 1, 2)
# print(y_test.shape)    # (10, 1088, 1, 2)
# print(y_val.shape)     # (8, 1088, 1, 2)

# StandardScaler를 하기 위해서 2차원으로 변환
x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1] * x_train.shape[2], x_train.shape[3])
x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1] * x_test.shape[2], x_test.shape[3])
x_val = x_val.reshape(x_val.shape[0] * x_val.shape[1] * x_val.shape[2], x_val.shape[3])
x_pred = x_pred.reshape(x_pred.shape[0] * x_pred.shape[1] * x_pred.shape[2], x_pred.shape[3])

# '0'인 데이터가 많기 때문에 StandardScaler로 스케일
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

# Conv1D 모델을 사용하기 위해서 3차원으로 변한
x_train = x_train.reshape(30 * 1088, 6, 5)
x_test = x_test.reshape(10 * 1088, 6, 5)
x_val = x_val.reshape(8 * 1088, 6, 5)
x_pred = x_pred.reshape(81 * 48, 6, 5)

y_train = y_train.reshape(30 * 1088, 1, 2)
y_test = y_test.reshape(10 * 1088, 1, 2)
y_val = y_val.reshape(8 * 1088, 1, 2)

##############################################################

'''
4. 변수 선택 및 모델 구축
Feature Engineering & Initial Modeling
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# 함수 : Quantile loss
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)


# 함수 : 모델링 Conv1D
def modeling() :
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same',\
         input_shape=(x_train.shape[1], x_train.shape[2])))                         # input (N, 5, 6)
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))
    model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'))

    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Reshape((1, 2)))                                                      # output (N, 1, 2)
    model.add(Dense(2))
    return model

##############################################################
'''
5. 모델 학습 및 검증
Model Tuning & Evaluation
'''
# 퀀타일 로스마다 최적의 loss 구하는 게 다를 듯???????????????????????????
quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# quantiles = [0.2, 0.4, 0.6, 0.7, 0.9]
# quantiles = [0.7, 0.9]

epoch = 50          # 훈련횟수
batch = 8           # batch size
loss_list = list()  # loss 기록

for q in quantiles :    # quatile loss 개수만큼 9번 반복
    print(f"\n>>>>>>>>>>>>>>>>>>>>>>  modeling start 'q_{q}'  >>>>>>>>>>>>>>>>>>>>>>") 

    model = modeling()                                                                                                  # Modeling
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')                     # Compile
    
    cp_save = f'../data/modelcheckpoint/solar_0126_s8_q_{q:.1f}.hdf5'                                                     # ModelCheckPoint 저장할 경로 지정
    es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')                                                    # 가장 적은 val_loss 구간에 멈출 수 있는 EarlyStopping
    cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, mode='min')                         # ModelCheckPoint 저장
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, verbose=1)                                      # learning rate 개선 없으면 0.4비율 씩 감소

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val), callbacks=[es, cp, lr]) # Train

    loss = model.evaluate(x_test, y_test,batch_size=8)                                     # Evaluate
    loss_list.append(loss)                                                                 # quatile loss 리스트에 저장

    y_pred = model.predict(x_pred)                                                         # Predict
    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])) # (3888, 2)
    y_pred = pd.concat([y_pred], axis=1)                                                   # 각 quantile loss 마다 예측한 값을 합친다.
    y_pred[y_pred<0] = 0                                                                   # 0이하로 나온 값들은 모두 0으로 처리
    y_pred = y_pred.to_numpy()                                                             # numpy로 변환

                                                                                                  # submission 파일에 저장
    column_name = f'q_{q}'                                                                        # quatile loss에 해당하는 컬럼에 predict한 값을 넣는다.
    submission.loc[submission.id.str.contains("Day7"), column_name] = np.around(y_pred[:, 0],3)   # Day7 (3888, 9) : 소수점 셋째자리에서 반올림
    submission.loc[submission.id.str.contains("Day8"), column_name] = np.around(y_pred[:, 1],3)   # Day8 (3888, 9) : 소수점 셋째자리에서 반올림

'''
6. 결과 및 결언
Conclusion & Discussion
'''

loss_mean = sum(loss_list) / len(loss_list)     # 9개 quatile loss 평균
print("loss_mean : ", loss_mean)                # 평균 확인
print("loss_list : ", loss_list )

submission.to_csv('../data/DACON_0126/submission.csv', index=False)  # 최종 predict 값들을 submission.csv에 저장



x = []                          # 생성된 파일 5개 중 점수 가장 잘 나온 5개 파일 묶어 중간값을 추출한다.
for i in range(20,25):          # 5개의 파일 합치기
    df = pd.read_csv(f'../data/DACON_0126/submission/submission_{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)
# print(x.shape)  # (5, 7776, 9)

df = pd.read_csv(f'../data/DACON_0126/submission/submission_{i}.csv', index_col=0, header=0)
for i in range(7776):                                                                                       # 각 행마다 중간값을 확인한다.
    for j in range(9):                                                                                      # qunatile loss 9개 만큼 반복
        a = []
        for k in range(5):                                                                                  # 5개의 파일만큼 반복
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a) 
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')  # 5개 파일의 중앙값
        
y = pd.DataFrame(df, index = None, columns = None)      
y.to_csv('../data/DACON_0126/submission/submission25.csv') 
"""