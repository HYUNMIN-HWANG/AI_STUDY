# Pipeline

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

#1. DATA
dataset = load_iris()
x = dataset.data 
y = dataset.target 

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.8, shuffle=True, random_state=55)
kf = KFold(n_splits=5, shuffle=True, random_state=47)

#2. Modeling

# [1]
# parameters = [
#     {'model__n_estimators' : [10, 100, 1000], 'model__max_depth' : [5, 6, 8, 10]},
#     {'model__min_samples_leaf' : [2, 4, 6], 'model__n_jobs' : [1, 2, 4]}
# ]

# pipe = Pipeline([("scaler", MinMaxScaler()),("model", RandomForestClassifier())])
# model = RandomizedSearchCV(pipe, parameters, cv=kf)
# model.fit(x_train, y_train)
# results = model.score(x_test, y_test)
# print("model.score " , results)

# [2]
parameters = [
    {'randomforestclassifier__n_estimators' : [10, 100, 1000], 'randomforestclassifier__max_depth' : [5,6,8,10]},
    {'randomforestclassifier__min_samples_leaf' : [2,4,6], 'randomforestclassifier__n_jobs' : [1,2,4]}
]

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
model = RandomizedSearchCV(pipe, parameters, cv=kf)
model.fit(x_train, y_train)
results = model.score(x_test, y_test)
print("model.score : ", results)



