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

##############################################################

loss_list = list()

epoch = 40
batch = 16

for q in quantiles :
    print(f"\n>>>>>>>>>>>>>>>>>>>>>> modeling start 'q_{q}'  >>>>>>>>>>>>>>>>>>>>>>") 

    #2. Modeling
    cp_load = f'../data/DACON_cp/save/solar_0132_q_{q:.1f}.hdf5'
    model = load_model(cp_load, compile = False)
    model.compile(loss = lambda y_true,y_pred: quantile_loss(q, y_true,y_pred), optimizer = 'adam')  # Compile                                        

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


loss_mean = sum(loss_list) / len(loss_list)     # 9개 quatile loss 평균
print("loss_mean : ", loss_mean)                # 평균 확인
print("loss_list : ", loss_list )

print(submission[15:20])
'''
                  id   q_0.1   q_0.2   q_0.3      q_0.4   q_0.5      q_0.6   q_0.7   q_0.8      q_0.9
15  0.csv_Day7_7h30m   0.000   0.000   0.000   0.000000   0.000   1.117000   0.823   0.724   1.574000
16  0.csv_Day7_8h00m   0.000   2.916   0.381   5.110000   9.954   7.217000   7.913   8.469   8.377000
17  0.csv_Day7_8h30m   2.336   6.363   4.010  10.530000  11.062  12.610000  13.619  13.584  13.635000
18  0.csv_Day7_9h00m   8.087   5.523  14.874  12.120000  17.451  19.785999  19.834  22.358  22.204000
19  0.csv_Day7_9h30m  12.367  17.952  20.337  21.056999  21.997  24.646000  26.076  27.034  28.082001
'''
submission.to_csv('../data/DACON_0126/submission_0126_4.csv', index=False)  # 최종 predict 값들을 submission.csv에 저장



'''
0125_7
loss_mean :  2.2480476101239524
loss_list :  [1.5606716871261597, 2.550719738006592, 2.7861902713775635, 3.1118435859680176, 2.980024814605713, 2.5997557640075684, 2.132758378982544, 1.6240900754928589, 0.8863741755485535]
                  id   q_0.1   q_0.2   q_0.3   q_0.4      q_0.5      q_0.6      q_0.7      q_0.8   q_0.9
15  0.csv_Day7_7h30m   0.000   0.000   0.332   0.000   0.591000   0.000000   0.610000   0.834000   0.415
16  0.csv_Day7_8h00m   0.552   3.328   3.280   2.650   5.749000   4.990000   7.802000   7.804000   8.451
17  0.csv_Day7_8h30m   1.756   6.025   7.940   8.488  10.228000  12.278000  13.102000  13.768000  13.293
18  0.csv_Day7_9h00m  11.019   8.941  14.307  13.013  14.169000  20.070999  20.952999  21.816000  22.848
19  0.csv_Day7_9h30m  12.739  14.550  15.266  21.372  21.358999  26.267000  26.427999  27.077999  27.230

0125_99
loss_mean :  2.2525884442859225
loss_list :  [1.4913822412490845, 2.346447467803955, 2.794707775115967, 3.1805944442749023, 3.0747148990631104, 2.4823317527770996, 2.248973846435547, 1.6435582637786865, 1.0105853080749512]
                  id   q_0.1   q_0.2      q_0.3   q_0.4   q_0.5      q_0.6      q_0.7      q_0.8      q_0.9
15  0.csv_Day7_7h30m   0.000   0.675   0.000000   0.918   1.716   1.555000   2.144000   2.646000   2.625000
16  0.csv_Day7_8h00m   2.870   1.588   2.659000   4.167   6.673   9.029000   8.311000  10.494000   9.695000
17  0.csv_Day7_8h30m   2.405   7.039   7.198000  10.644  10.230  13.519000  14.792000  16.034000  15.389000
18  0.csv_Day7_9h00m   3.852  10.566  19.132999  16.500  15.929  20.580999  22.445999  22.594000  23.490000
19  0.csv_Day7_9h30m  11.933  12.674  16.368000  20.605  19.125  24.556000  26.646999  27.033001  28.134001

0126_3
loss_mean :  2.038280506928762
loss_list :  [1.510628342628479, 1.9271990060806274, 1.9599796533584595, 2.689218759536743, 2.872896909713745, 2.6089110374450684, 2.14351487159729, 1.6339685916900635, 0.9982073903083801]
                  id   q_0.1   q_0.2   q_0.3      q_0.4   q_0.5      q_0.6   q_0.7   q_0.8      q_0.9
15  0.csv_Day7_7h30m   0.000   0.000   0.000   0.000000   0.000   1.117000   0.823   0.724   1.574000
16  0.csv_Day7_8h00m   0.000   2.916   0.381   5.110000   9.954   7.217000   7.913   8.469   8.377000
17  0.csv_Day7_8h30m   2.336   6.363   4.010  10.530000  11.062  12.610000  13.619  13.584  13.635000
18  0.csv_Day7_9h00m   8.087   5.523  14.874  12.120000  17.451  19.785999  19.834  22.358  22.204000
19  0.csv_Day7_9h30m  12.367  17.952  20.337  21.056999  21.997  24.646000  26.076  27.034  28.082001

0126_4
loss_mean :  2.176380349530114
loss_list :  [1.488850474357605, 2.1493539810180664, 2.6711742877960205, 2.981830358505249, 2.8930768966674805, 2.6786386966705322, 2.152386426925659, 1.6095972061157227, 0.9625148177146912]
                  id   q_0.1  q_0.2   q_0.3      q_0.4      q_0.5      q_0.6      q_0.7   q_0.8   q_0.9
15  0.csv_Day7_7h30m   0.000  0.000   0.000   0.000000   0.000000   1.891000   1.422000   0.484   1.953
16  0.csv_Day7_8h00m   0.682  1.471   4.662   3.470000   6.354000   8.197000   7.555000   8.317   9.819
17  0.csv_Day7_8h30m   1.790  3.242   8.281  11.885000  10.382000  13.269000  13.061000  13.148  14.337
18  0.csv_Day7_9h00m  11.215  5.311   5.919  14.086000  16.455999  20.184000  21.096001  21.295  22.504
19  0.csv_Day7_9h30m   7.127  7.816  18.545  20.650999  21.851999  25.393999  26.246000  25.848  27.329
'''