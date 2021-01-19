import pandas as pd
import numpy as np
import os
import glob
import random

import warnings
warnings.filterwarnings("ignore")

# train 데이터 불러오기 >> x_train
train = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.tail())

# submission 데이터 불러오기 
submission = pd.read_csv('../data/DACON_0126/sample_submission.csv')
# print(submission.tail())

# 데이터 자르기
def preprocess_data (data, is_train=True) :
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train == True :    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날 TARGET을 붙인다.
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날 TARGET을 붙인다.
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 이틀치 데이터만 빼고 전체
    elif is_train == False :         
        # Day, Minute 컬럼 제거
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :] # 마지막 하루치 데이터

df_train = preprocess_data(train)
# print(df_train.shape)   # (52464, 9)
# print(df_train.iloc[:48])
# print(df_train.iloc[48:96])

# test 데이터 불러오기 >> x_pred
df_pred = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp, is_train=False)    # Day, Minute 컬럼 삭제
    df_pred.append(temp)

X_pred = pd.concat(df_pred)
# print(X_pred.shape) # (3888, 7)
# print(X_pred.head())
# print(X_pred.tail())

from sklearn.model_selection import train_test_split
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)   # y : 다음날
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)   # y : 다다음날
# print(X_train_1.head())
# print(Y_train_1.head())
# print(X_pred.head())

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor
# Get the model and the predictions in (a) - (b)
def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_pred):
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.027, subsample=0.7)
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=300, verbose=500)
         
    # (b) Predictions
    pred = pd.Series(model.predict(X_pred).round(2))
    return model, pred

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
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_pred)
print(results_1.sort_index()[:48])

# Target2
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_pred)
results_2.sort_index()[:48]

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
print(submission)

submission.to_csv('../data/DACON_0126/submission/submission_v3.csv', index=False)
