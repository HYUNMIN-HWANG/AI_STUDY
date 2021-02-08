'''
[Dacon] AI프렌즈 시즌3 SMP 및 전력수요 예측 경진대회
팀명 : xian
제출날짜 : 2020년 2월 1일
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
print(train[:7])
'''
   Day  Hour  Minute  DHI  DNI   WS     RH   T  TARGET
0    0     0       0    0    0  1.5  69.08 -12     0.0
1    0     0      30    0    0  1.5  69.06 -12     0.0
2    0     1       0    0    0  1.6  71.78 -12     0.0
3    0     1      30    0    0  1.6  71.75 -12     0.0
4    0     2       0    0    0  1.6  75.20 -12     0.0
5    0     2      30    0    0  1.5  69.29 -11     0.0
6    0     3       0    0    0  1.5  72.56 -11     0.0
'''

# submission 파일 불러오기
submission = pd.read_csv('../data/DACON_0126/sample_submission.csv')
# print(submission.shape) # (7776, 10)
print(submission[:7])
'''
                 id  q_0.1  q_0.2  q_0.3  q_0.4  q_0.5  q_0.6  q_0.7  q_0.8  q_0.9
0  0.csv_Day7_0h00m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
1  0.csv_Day7_0h30m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
2  0.csv_Day7_1h00m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
3  0.csv_Day7_1h30m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
4  0.csv_Day7_2h00m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
5  0.csv_Day7_2h30m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
6  0.csv_Day7_3h00m    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0
'''

##############################################################

#1. DATA

# 함수 : 태양열을 더 잘 예측할 수 있는 GHI, T-Td column 추가
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

# 함수 : train/test data column 전처리
# Target과 상관계수가 높은 칼럼만 사용한다.
# 예측하고자 하는 다음 이틀 날의 칼럼을 추가한다.
def preprocess_data(data, is_train=True):
    data = Add_features(data)
    temp = data.copy()
    temp = temp[['Day','TARGET','GHI','DHI','DNI','T-Td']]                   

    if is_train==True:       # train data                                                        
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') 
        temp = temp.dropna()                                                
        return temp.iloc[:-96, :]                                       

    elif is_train==False:     # test data                                       
        return temp.iloc[-48*6:, 1:]

#함수 : 같은 시간대끼리 모으기
def same_train(train) :
    temp = train.copy()
    x = list()                                      
    final_x = list()                                
    for i in range(48) :                            # 48개의 시간대 만큼 반복
        same_time = pd.DataFrame()                  
        for j in range(int(len(temp)/48)) :        
            tmp = temp.iloc[i + 48*j, : ]           # 같은 시간대에 있는 행만 모으기
            tmp = tmp.to_numpy()
            tmp = tmp.reshape(1, tmp.shape[0])      
            tmp = pd.DataFrame(tmp)
            same_time = pd.concat([same_time, tmp]) 
        x = same_time.to_numpy()    
        final_x.append(x)                        
    return np.array(final_x) 


# 함수 : 시계열 데이터로 자르기
def split_xy(dataset, time_steps) :          
    x, y = list(), list()
    for i in range(len(dataset)) :          
        x_end = i + time_steps            
        y_end = x_end-1                      
        if x_end > len(dataset) :        
            break
        tmp_x = dataset[i : x_end, 1:-2]      
        tmp_y = dataset[y_end, -2:]         
        x.append(tmp_x)                    
        y.append(tmp_y)                  
    return np.array(x), np.array(y)   

##############################################################

'''
2. 데이터 전처리
Data Cleansing & Pre-Processing
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
# 상관계수 0.6 이상인 컬럼만 사용함 : Target, GHI, DHI, DNI, T-Td

# 같은 시간대 별로 묶기
same_time = same_train(df_train)
print(same_time.shape)  # (48, 1093, 8)
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

# x와 y를 시계열 데이터로 자르기
x, y = list(), list()
for i in range(48):
    tmp1,tmp2 = split_xy(same_time[i], 6)
    x.append(tmp1)
    y.append(tmp2)

x = np.array(x)
print("x.shape : ", x.shape)    #   (48, 1088, 6, 5)

y = np.array(y)
y = y.reshape(48, 1088, 1, 2)
print("y.shape : ", y.shape)    #   (48, 1088, 1, 2)


print(x[0,:3,:,:])
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
  [0.         0.         0.         0.         7.88527903]]]
'''
print(x[0,-3:,:,:])
'''
[[[0.         0.         0.         0.         4.78875474]
  [0.         0.         0.         0.         7.06061247]
  [0.         0.         0.         0.         5.26179355]
  [0.         0.         0.         0.         6.46620945]
  [0.         0.         0.         0.         3.17928125]
  [0.         0.         0.         0.         2.78822817]]

 [[0.         0.         0.         0.         7.06061247]
  [0.         0.         0.         0.         5.26179355]
  [0.         0.         0.         0.         6.46620945]
  [0.         0.         0.         0.         3.17928125]
  [0.         0.         0.         0.         2.78822817]
  [0.         0.         0.         0.         8.48614485]]

 [[0.         0.         0.         0.         5.26179355]
  [0.         0.         0.         0.         6.46620945]
  [0.         0.         0.         0.         3.17928125]
  [0.         0.         0.         0.         2.78822817]
  [0.         0.         0.         0.         8.48614485]
  [0.         0.         0.         0.         9.00927341]]]
'''
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

# print(y_train.shape)    # (30, 1088, 1, 2)
# print(y_test.shape)     # (10, 1088, 1, 2)
# print(y_val.shape)      # (8, 1088, 1, 2)

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
3. 변수 선택 및 모델 구축
Feature Engineering & Initial Modeling
'''

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K

# 함수 : Modeling
# quantile loss 마다 적절한 모델을 구성
def modeling(q) :
    if q == 0.1 :
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))      # # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40             
        batch = 8           
        return model, epoch, batch

    elif q == 0.2 :
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))


        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40           
        batch = 16            
        return model, epoch, batch

    elif q == 0.3 :
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40          
        batch = 8
        return model, epoch, batch

    elif q == 0.4 :
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40           
        batch = 8     
        return model, epoch, batch

    elif q == 0.5 :
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(8))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40           
        batch = 4       
        return model, epoch, batch

    elif q == 0.6 :
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))
 
        epoch = 40           
        batch = 8          
        return model, epoch, batch

    elif q == 0.7 :
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40            
        batch = 16           
        return model, epoch, batch

    elif q == 0.8 :
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(32))
        model.add(Dense(16))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40            
        batch = 8           
        return model, epoch, batch

    elif q == 0.9 :               
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same',\
            input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 6, 5)
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu', padding='same'))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='relu'))
        model.add(Reshape((1, 2)))  # output (N, 1, 2)
        model.add(Dense(2))

        epoch = 40             
        batch = 4
        return model, epoch, batch

##############################################################
'''
4. 모델 학습 및 검증
Model Tuning & Evaluation
'''
# 함수 : Quantile loss
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
quantiles = [ 0.6, 0.8]

from tensorflow.keras.optimizers import Adam

loss_list = []         # loss 기록

for q in quantiles :    # quatile loss 개수만큼 9번 반복
    print(f"\n>>>>>>>>>>>>>>>>>>>>>>  modeling start 'q_{q}'  >>>>>>>>>>>>>>>>>>>>>>") 

    # optimizer = Adam(lr=0.1)
    model, epoch, batch = modeling(q)  # Modeling                                                                                               
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')  # Compile                   

    cp_save = f'../data/DACON_cp/solar_{q:.1f}.hdf5'                                                   
    es = EarlyStopping(monitor='val_loss', patience=20, mode='auto')                                                   
    cp = ModelCheckpoint(filepath=cp_save, monitor='val_loss', save_best_only=True, mode='min')                 
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.4, verbose=1)                              

    model.fit(x_train, y_train, epochs=epoch, batch_size=batch, validation_data=(x_val, y_val), callbacks=[es, cp, lr]) # Train

    loss = model.evaluate(x_test, y_test,batch_size=batch)  # Evaluate 
    loss_list.append(loss)                                   

    y_pred = model.predict(x_pred)  # Predict                                        
    y_pred = pd.DataFrame(y_pred.reshape(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])) # (3888, 2)
    y_pred = pd.concat([y_pred], axis=1)                                           
    y_pred[y_pred<0] = 0                                                           
    y_pred = y_pred.to_numpy()                                              

                                                                                                  
    column_name = f'q_{q}'     # submission 파일에 저장                                                            
    submission.loc[submission.id.str.contains("Day7"), column_name] = np.around(y_pred[:, 0],3)   
    submission.loc[submission.id.str.contains("Day8"), column_name] = np.around(y_pred[:, 1],3)


'''
5. 결과 및 결언
Conclusion & Discussion
'''

loss_mean = sum(loss_list) / len(loss_list)     # 9개 quatile loss 평균
print("loss_mean : ", loss_mean)                # 평균 확인
print("loss_list : ", loss_list )

print(submission[15:20])
'''
                  id  q_0.1  q_0.2   q_0.3      q_0.4   q_0.5  q_0.6      q_0.7      q_0.8      q_0.9
15  0.csv_Day7_7h30m  0.000  0.000   0.000   0.000000   0.000  5.622   0.881000  36.161999   3.228000
16  0.csv_Day7_8h00m  1.675  1.277   2.854   2.583000   5.706  5.622   9.069000  36.161999   8.110000
17  0.csv_Day7_8h30m  1.008  2.596   7.605   9.897000  11.003  5.622  15.264000  36.161999  15.540000
18  0.csv_Day7_9h00m  3.889  5.781  11.846  16.906000  18.254  5.622  23.141001  36.161999  26.207001
19  0.csv_Day7_9h30m  6.926  6.785  14.275  19.757999  21.577  5.622  26.638000  36.161999  31.260000
'''
submission.to_csv('../data/DACON_0126/submission_0131_00.csv', index=False)  # 최종 predict 값들을 submission.csv에 저장

'''
[결언]

비전공자로 인공지능을 공부한지 얼마 안된 채 대회에 참가해서 처음에는 막막하기도 하고 답답하기도 했는데요. 
그래도 지금까지 제가 배운 내용들을 활용해가면서 문제를 풀 수 있어서 재밌었습니다. 
이런 AI 대회에 참가할 수 있는 것만으로도 기뻤는데 상위권까지 오르다니 더욱 행복했습니다.
이번 대회의 경우 초반에 데이터를 어떻게 시계열 데이터로 구현할 것인지가 가장 힘들었는데 
토론 게시판에 있는 베이스라인을 보면서 감을 익혔던 것 같습니다.
아직 미숙한 실력이기에 앞으로 인공지능을 더 많이 배워서 또 다른 대회에 도전해보고 싶습니다! 
좋은 대회 열어주셔서 감사합니다.
'''
