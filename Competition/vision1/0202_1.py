# PCA
# XGboost
# 실패

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D

from xgboost import XGBClassifier, plot_importance

######################################################

# 데이터 로드
train = pd.read_csv('../data/DACON_vision1/train.csv')
print(train.shape)  # (2048, 787)

submission = pd.read_csv('../data/DACON_vision1/submission.csv')
print(submission.shape) # (20480, 2)

pred = pd.read_csv('../data/DACON_vision1/test.csv')
print(pred.shape)   # (20480, 786)

######################################################


#1. DATA
# x
x = train.drop(['id', 'digit', 'letter'], axis=1).values
x = x/255.0
# x = x.reshape(-1, 28, 28, 1)
print(x.shape)  # (2048, 28, 28, 1) >> (2048, 784)

# y
y_tmp = train['digit']
y = np.zeros((len(y_tmp), len(y_tmp.unique()))) # np.zeros(shape, dtype, order) >> 0으로 초기화된 넘파이 배열 
for i, digit in enumerate(y_tmp) :
    y[i, digit] = 1
print(y.shape)  # (2048, 10)

# predict
x_pred = pred.drop(['id', 'letter'], axis=1).values
# x_pred = x_pred.reshape(-1, 28, 28, 1)
x_pred = x_pred/255.0 
print(x_pred.shape) # (20480, 28, 28, 1) >> (20480, 784)

# 컬럼 압축
# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)
# d = np.argmax(cumsum >= 0.95)+1
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)    # d :  98
pca = PCA(n_components=98)
x2 = pca.fit_transform(x)
x2_pred = pca.fit_transform(x_pred)

print(x2.shape) # (2048, 98)

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=47)
print(x_train.shape, x_test.shape)     # (1638, 98) (410, 98)
print(y_train.shape, y_test.shape)     # (1638, 10) (410, 10)
kf = KFold(n_splits=7, shuffle=True, random_state=33)

#2. Modeling
# parameters = [
#     {"n_estimators":[90, 100, 200], "learning_rate":[0.01, 0.001]},
#     {"n_estimators":[90, 100, 200], "learning_rate":[0.01, 0.001], "max_depth":[4, 5, 6]},
#     {"n_estimators":[90, 100, 200], "learning_rate":[0.01, 0.001], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9]},
#     {"n_estimators":[90, 100, 200], "learning_rate":[0.01, 0.001], "max_depth":[4, 5, 6], "colsample_bytree":[0.6, 0.9], "colsample_bylevel" :[0.6, 0.7, 0.9]}
# ]

# model = RandomizedSearchCV\
#     (XGBClassifier(n_jobs=8, use_label_encoder=False, n_estimators=20,  learning_rate=0.01),\
#         parameters, cv=kf)
# model = XGBClassifier(n_jobs=8, use_label_encoder=False, n_estimators=20,  learning_rate=0.01)
model = RandomForestClassifier()

#3. Train
# model.fit(x_train, y_train, eval_metric='mlogloss', verbose=True, \
#     eval_set=[(x_train, y_train), (x_test, y_test)],
#      early_stopping_rounds=10)

model.fit(x_train, y_train)

#4. Score, Predict
# print("최적의 매개변수 : ", model.best_estimator_)

score = model.score(x_test, y_test)
print("score : ", score)    # score :  0.0

y_pred = model.predict(x2_pred)
acc = accuracy_score(y_test, y_pred)
print("acc :",acc)

