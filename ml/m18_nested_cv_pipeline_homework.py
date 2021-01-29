# 과제 :
# model : RandomForest를 사용
# 파이프라인 엮어서 25번 돌리기
# load_diabetes

import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')
import pandas as pd 

###########################################################

#1. DATA
dataset = load_diabetes()
x = dataset.data 
y = dataset.target 
# print(x.shape, y.shape)

# preprocessing >>  K-Fold 
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

#2. Modeling
# pipline : 파라미터튜닝에 전처리까지 합친다. >> 전처리와 모델을 합친다.

# # [1] Pipeline
# parameters=[
#     {'model__n_estimators' : [100, 200, 300], 'model__max_depth' : [6, 8, 10, 12]},
#     {'model__max_depth' : [6, 8, 10, 12], 'model__min_samples_leaf' : [3, 7, 10]},
#     {'model__min_samples_split' : [2, 3, 5, 9], 'model__n_jobs' : [-1, 2, 4]}
# ]

# # [2] make_pipeline
# parameters=[
#     {'randomforestregressor__n_estimators' : [100, 200], 'randomforestregressor__max_depth' : [6, 8, 10, 12]},
#     {'randomforestregressor__max_depth' : [6, 8, 10, 12], 'randomforestregressor__min_samples_leaf' : [3, 7, 10]},
#     {'randomforestregressor__min_samples_split' : [2, 3, 5, 9], 'randomforestregressor__n_jobs' : [-1, 2, 4]}
# ]

for train_index, test_index in kfold.split(x) :
    # print(train_index,"\n")
    # print(test_index,"\n")
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestRegressor())])
    # model = RandomizedSearchCV(pipe, parameters, cv=kfold)
    score = cross_val_score(model, x_train, y_train, cv=kfold)

    print('교차검증점수 : ', score, "\n")




# GridSearch
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=10)
# 최종정답률 0.4838812611290275
# aaa  0.4838812611290275

# RandomSearch
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=10, min_samples_split=10)
# 최종정답률 0.47678641072030825
# aaa  0.47678641072030825

# pipeline (MinMaxscaler)
# model.score :  0.42556136196707384

# pipeline(Standardscaler)
# model.score:  0.41152922668231506

# pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'>:0.48080011996690997
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'>:0.46004691837434464
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'>:0.4760492602051315
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'>:0.47452782842686003


# train / test/ validataion - pipeliine & randomforest
# 교차검증점수 :  [0.46245767 0.42880664 0.29264052 0.55223488 0.44054647]
# 교차검증점수 :  [0.18706624 0.48035615 0.46648626 0.43374131 0.43654188]
# 교차검증점수 :  [0.40966921 0.28273204 0.3791408  0.50513237 0.36888939]
# 교차검증점수 :  [0.39513525 0.46463413 0.36613272 0.37842838 0.34767955]
# 교차검증점수 :  [0.46137109 0.45764707 0.39580661 0.40758026 0.27694801]