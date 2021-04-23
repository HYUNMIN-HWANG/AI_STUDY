import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.shape)  # (52560, 9)
submission = pd.read_csv('../data/DACON_0126/sample_submission.csv')
# print(submission.shape) # (7776, 10)

#1. DATA

# train data column 정리
# 끝에 다음날, 다다음날 TARGET 데이터를 column을 추가한다.
def preprocess_data(data, is_train=True):

    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]

    if is_train==True:          
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날의 Target
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날의 Target
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 뒤에서 이틀은 뺀다. (예측하고자 하는 날짜이기 때문)

    elif is_train==False:
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]       
        return temp.iloc[-48:, :]   # 0 ~ 6일 중 마지막 6일 데이터만 남긴다. (6일 데이터로 7, 8일을 예측하고자 함) 

df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 9)
# print(df_train.columns) 
# Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1', 'Target2'], dtype='object')


# 81개의 0 ~ 7 Day 데이터 합치기
df_test = []

for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)
    df_test.append(temp)

X_test = pd.concat(df_test)
# print(X_test.shape) # (3888, 7)


#2. Modeling
from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

# print(X_train_1.shape)  # (36724, 7)
# print(X_valid_1.shape)  # (15740, 7)
# print(Y_train_1.shape)  # (36724,)
# print(Y_valid_1.shape)  # (15740,)

# quantile loss
# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
quantiles = [0.5, 0.6]

from lightgbm import LGBMRegressor

# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                            n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)                          
    # objective = 'quantile' >> quantile 회귀모델
    # n_estimators           >> (default=100) 훈련시킬 tree의 개수
    # bagging_fraction       >> 0 ~ 1 사이, 랜덤 샘플링
    # learning_rate          >> 일반적으로 0.01 ~ 0.1 사이
    # subsample              >> Row sampling, 즉 데이터를 일부 발췌해서 다양성을 높이는 방법으로 쓴다.
    # print("model : ", model)
    # 출력결과 >> model :  LGBMRegressor(alpha=0.5, bagging_fraction=0.7, learning_rate=0.027,
            #   n_estimators=10000, objective='quantile', subsample=0.7)

    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
            eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
    # early_stopping_rounds  >> validation셋에 더이상 발전이 없으면 그만두게 설정할때 이를 몇번동안 발전이 없으면 그만두게 할지 여부.
    
    # (b) Predictions
    pred = pd.Series(model.predict(X_test).round(2))    
    # Series : 1차원 배열
    # print(pred) : X_test로 predict한 결과가 나옴
    return pred, model


# Target 예측
def train_data(X_train, Y_train, X_valid, Y_valid, X_test):
    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print(q)
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        LGBM_models.append(model)
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles

    return LGBM_models, LGBM_actual_pred

# Target1
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)

# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)

# print(results_1.shape, results_2.shape) # (3888, 9) (3888, 9)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values

submission.to_csv('../data/DACON_0126/submission_LGBM_v1.csv', index=False) # score : 	2.0202123047	