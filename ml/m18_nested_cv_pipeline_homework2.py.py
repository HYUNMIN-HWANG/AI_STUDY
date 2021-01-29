# 과제 :
# model : RandomForest를 사용
# 파이프라인 엮어서 25번 돌리기
# load_wine

import numpy as np
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# 모델마다 나오는 결과 값을 비교한다.
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

import warnings
warnings.filterwarnings('ignore')
import pandas as pd 

###########################################################

#1. DATA
dataset = load_wine()
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
#     {'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth' : [6, 8, 10, 12]},
#     {'randomforestclassifier__max_depth' : [6, 8, 10, 12], 'randomforestclassifier__min_samples_leaf' : [3, 7, 10]},
#     {'randomforestclassifier__min_samples_split' : [2, 3, 5, 9], 'randomforestclassifier__n_jobs' : [-1, 2, 4]}
# ]

for train_index, test_index in kfold.split(x) :

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestClassifier())])
    # model = RandomizedSearchCV(pipe, parameters, cv=5)
    score = cross_val_score(model, x_train, y_train, cv=kfold )
    print('교차검증점수 : ', score)



# gridSearch
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_jobs=4)
# 최종정답률 0.9722222222222222
# aaa  0.9722222222222222

# RandomSearch
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, n_estimators=300, n_jobs=-1)
# 최종정답률 0.9722222222222222
# aaa  0.9722222222222222

# pipeline (MinMaxscaler)
# model.score :  0.9722222222222222

# pipeline (StandardScaler)
# model.score :  0.9722222222222222

# pipeline
# MinMaxScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'>:0.9722222222222222
# MinMaxScaler() <class 'sklearn.model_selection._search.GridSearchCV'>:0.9722222222222222
# StandardScaler() <class 'sklearn.model_selection._search.RandomizedSearchCV'>:0.9722222222222222
# StandardScaler() <class 'sklearn.model_selection._search.GridSearchCV'>:0.9444444444444444

# train / test/ validataion - pipeliine & randomforest
# 교차검증점수 :  [1.         1.         1.         0.96428571 1.        ]
# 교차검증점수 :  [0.96551724 1.         0.96428571 0.92857143 1.        ]
# 교차검증점수 :  [1.         0.96551724 1.         0.92857143 1.        ]
# 교차검증점수 :  [1.         1.         0.96551724 0.96428571 0.96428571]
# 교차검증점수 :  [1.         0.96551724 0.96551724 0.96428571 1.        ]

