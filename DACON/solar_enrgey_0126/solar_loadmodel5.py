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
    c = 243.12
    b = 17.62
    gamma = (b * (data['T']) / (c + (data['T']))) + np.log(data['RH'] / 100)
    dp = ( c * gamma) / (b - gamma)
    data.insert(1,'Td',dp)
    data.insert(1,'T-Td',data['T']-data['Td'])
    return data

# 함수 : train data column 정리
# 끝에 다음날, 다다음날 TARGET 데이터 column을 추가한다.
def preprocess_data(data, is_train=True):
    data = Add_features(data)
    # print(data.columns) 
    # Index(['Day', 'T-Td', 'Td', 'GHI', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH','T', 'TARGET'], dtype='object')
    temp = data.copy()
    temp = temp[['Day','TARGET','GHI','DHI','DNI','T-Td']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96, :]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train==False:     
        return temp.iloc[-48*6:, 1:]  # 6일치만 사용

#함수 : 같은 시간대끼리 모으기
def same_train(train) :
    temp = train.copy()
    x = list()
    final_x = list()
    for i in range(48) :
        same_time = pd.DataFrame()
        for j in range(int(len(temp)/48)) :
            tmp = temp.iloc[i + 48*j, : ]
            tmp = tmp.to_numpy()
            tmp = tmp.reshape(1, tmp.shape[0])
            tmp = pd.DataFrame(tmp)
            # print(tmp)
            same_time = pd.concat([same_time, tmp])
        x = same_time.to_numpy()
        final_x.append(x)
    return np.array(final_x)

# print(len(train)) # 52560
# print(same_train(train).shape) # (48, 1095, 9)

# 함수 : 시계열 데이터로 자르기 (x는 6행씩, y는 1행씩)
def split_xy(dataset, time_steps) :  # data, 6
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end = i + time_steps
        y_end = x_end-1
        if x_end > len(dataset) :
            break
        tmp_x = dataset[i : x_end, 1:-2]      # ['TARGET', 'GHI', 'DHI', 'DNI', 'T-Td']
        tmp_y = dataset[y_end, -2:]           # ['Target1', 'Target2']
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

##############################################################

# train data
df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 8)
# print(df_train.columns) 
# Index(['Day', 'TARGET', 'GHI', 'DHI', 'DNI', 'T-Td', 'Target1', 'Target2'], dtype='object')


# 상관계수 확인
# print(df_train.corr())
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(font_scale=1.0, font='Malgun Gothic', rc={'axes.unicode_minus':False}) 
# sns.heatmap(data=df_train.corr(),square=True, annot=True, cbar=True)
# plt.show()
# > 기준 : Target1, Target2
# > 상관계수 0.6 이상 : Target, GHI, DHI, DNI, T-Td

# 같은 시간대 별로 묶기
same_time = same_train(df_train)
# print(same_time.shape)  # (48, 1093, 8)
# print(same_time[0:3, :5 :])

# X = same_time.to_numpy()
# print(X.shape)      # (52464, 8)

x, y = list(), list()
for i in range(48):
    tmp1,tmp2 = split_xy(same_time[i], 6)
    x.append(tmp1)
    y.append(tmp2)

x = np.array(x)
y = np.array(y)
print("x.shape : ", x.shape) # (48, 1088, 6, 5)
print("y.shape : ", y.shape) # (48, 1088, 2)

y = y.reshape(48, 1088, 1, 2)


# test data : 81개의 0 ~ 7 Day 데이터 합치기
df_test = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    # print(temp.columns) # Index(['TARGET', 'GHI', 'DHI', 'DNI', 'T-Td'], dtype='object')
    temp = pd.DataFrame(temp)
    temp = same_train(temp)
    df_test.append(temp)

x_pred = np.array(df_test)
print("x_pred.shape : ", x_pred.shape) # (81, 48, 6, 5)


##############################################################
# x >> preprocessing
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, shuffle=True, random_state=32)
x_train, x_val, y_train, y_val, = \
    train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=32)

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

# StandardScaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)
x_pred = scaler.transform(x_pred)

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

#2. Modeling
# def modeling() :
#     model = Sequential()
#     model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same',\
#          input_shape=(x_train.shape[1], x_train.shape[2]))) # input (N, 336, 6)
#     model.add(Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'))
#     model.add(Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
#     model.add(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))

#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(96, activation='relu'))
#     model.add(Reshape((48,2)))  # output (N, 48, 2)
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(2))
#     return model

##############################################################

loss_list = list()

for q in quantiles :
    print(f"\n>>>>>>>>>>>>>>>>>>>>>> modeling start 'q_{q}'  >>>>>>>>>>>>>>>>>>>>>>") 

    #2. Modeling
    # model = modeling()
    cp_load = f'../data/modelcheckpoint/solar_0126_s99_q_{q:.1f}.hdf5'
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
    result = model.evaluate(x_test, y_test, batch_size=8)
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

    # submission
    # column_name = 'q_' + str(q)
    column_name = f'q_{q}'
    submission.loc[submission.id.str.contains("Day7"), column_name] = y_pred[:, 0].round(2)  # Day7 (3888, 9)
    submission.loc[submission.id.str.contains("Day8"), column_name] = y_pred[:, 1].round(2)   # Day8 (3888, 9)


loss_mean = sum(loss_list) / len(loss_list) # 9개 loss 평균
print("loss_mean : ", loss_mean)    #
print("9 loss : ", loss_list)

# to csv
submission.to_csv('../data/DACON_0126/submission_0126_99.csv', index=False)  # score : 

# 01-25
# submission 1
# loss_mean :  2.0906164050102234
# 9 loss :  [1.401052713394165, 2.3083579540252686, 2.5304408073425293, 2.8630199432373047, 2.846902847290039, 2.3841283321380615, 2.1714553833007812, 1.492830514907837, 0.8173591494560242]
# score 2.1031931547	

# submission 2
# loss_mean :  2.34127891725964
# 9 loss :  [1.5960848331451416, 2.581838607788086, 3.0542895793914795, 3.261608123779297, 3.130524158477783, 2.7902138233184814, 2.245457172393799, 1.5715806484222412, 0.8399133086204529]
# score : 1.9817587091	


# submission 3
# loss_mean : 2.3986232611868115
# 9 loss :  [1.6318514347076416, 2.5639123916625977, 3.103139638900757, 3.417604446411133, 3.243687868118286, 2.8485240936279297, 2.2717182636260986, 1.6412888765335083, 0.8658823370933533]
# score : 1.9416983628	

# 1
# loss_mean :  2.4379326038890414
# 9 loss :  [1.6375917196273804, 2.6726913452148438, 3.2661657333374023, 3.39941143989563, 3.2425382137298584, 2.865349292755127, 2.3362159729003906, 1.648589849472046, 0.8728398680686951]
# score : 1.9442525224

##############################
# 01_26

# 5
# loss_mean :  2.237951305177477
# 9 loss :  [1.439159870147705, 2.3633313179016113, 2.7087929248809814, 3.21466326713562, 3.023529529571533, 2.649862051010132, 2.2199928760528564, 1.616608738899231, 0.9056211709976196]
# score : 1.9638152833

# 7
# loss_mean :  2.3000545501708984
# 9 loss :  [1.4710476398468018, 2.528590440750122, 2.85558819770813, 3.2215945720672607, 3.13633131980896, 2.723604440689087, 2.260201930999756, 1.6264822483062744, 0.8770501613616943]

# 2
# loss_mean :  2.3534387747446694
# 9 loss :  [1.5558990240097046, 2.550999641418457, 3.055598020553589, 3.1890032291412354, 3.181591510772705, 2.871227741241455, 2.268519878387451, 1.6338671445846558, 0.8742427825927734]

# 3
# loss_mean :  2.124546832508511
# 9 loss :  [1.4475194215774536, 2.0000557899475098, 2.234090566635132, 2.8745200634002686, 2.9738996028900146, 2.8361709117889404, 2.2869114875793457, 1.5982439517974854, 0.8695096969604492]

# 4
# loss_mean :  2.2341311242845325
# 9 loss :  [1.550812005996704, 2.5161798000335693, 2.957852602005005, 3.1197245121002197, 2.9248976707458496, 2.6380538940429688, 2.115185499191284, 1.4894014596939087, 0.7950726747512817]
	
# 6
# loss_mean :  2.0745634105470447
# 9 loss :  [1.36246657371521, 1.875643253326416, 2.647616147994995, 2.416485071182251, 2.8553872108459473, 2.6726796627044678, 2.2227296829223633, 1.6835393905639648, 0.9345237016677856]

# 7
# loss_mean :  2.161611404683855
# 9 loss :  [1.3758916854858398, 2.285569429397583, 2.5791571140289307, 3.1202309131622314, 2.8802123069763184, 2.6057827472686768, 2.1176722049713135, 1.603780746459961, 0.8862054944038391]

# 8
# loss_mean :  2.2191689014434814
# 9 loss :  [1.3472135066986084, 2.3323566913604736, 2.8718361854553223, 3.1279642581939697, 2.8574719429016113, 2.67148756980896, 2.186594009399414, 1.6787039041519165, 0.8988920450210571]


# submission 3
# loss_mean :  2.0315550433264837
# 9 loss :  [1.3472135066986084, 1.875643253326416, 2.7838239669799805, 2.416485071182251, 2.8553872108459473, 2.6057827472686768, 2.115185499191284, 1.4894014596939087, 0.7950726747512817]
# 2.0617842035	

# 99
