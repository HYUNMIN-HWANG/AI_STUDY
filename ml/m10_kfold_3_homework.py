# homework
# train, test 나눈 다음에 train만 validation 하지 말고,
# kfold 한 후에 >> train_test_split 사용

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

########################################################

#1. DATA
dataset = load_iris()
x = dataset.data 
y = dataset.target 

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )

# x preprocessing >>  K-Fold 
kfold = KFold(n_splits=5, shuffle=True) # 데이터를 5등분한다. > train data 와 test data 로 구분한다.

#2. Modeling
model=[LinearSVC(), SVC(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier()]

# kfold 만 적용
# for algorithm in model :
#     score = cross_val_score(algorithm, x, y, cv=kfold)
    # print('score : ', score, '-'+str(algorithm))

# score :  [1.         1.         0.93333333 0.9        0.93333333] -LinearSVC()
# score :  [0.96666667 1.         1.         0.86666667 0.93333333] -SVC()
# score :  [0.9        0.96666667 0.96666667 0.96666667 0.96666667] -KNeighborsClassifier()
# score :  [0.96666667 0.96666667 1.         0.96666667 0.86666667] -LogisticRegression()
# score :  [0.96666667 0.96666667 0.93333333 0.96666667 0.93333333] -RandomForestClassifier()
# score :  [0.93333333 0.93333333 0.96666667 0.83333333 0.96666667] -DecisionTreeClassifier()

# kfold >> train, test 분리
for train_index, test_index in kfold.split(x) :
    
    # print("TRAIN", train_index, '\n', "TEST", test_index)
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # train_test_split >> train, validation 분리
    x_train, x_val, y_train, y_val = \
        train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=47)
    # print(x.shape)          # (150, 4)
    # print(x_train.shape)    # (96, 4)
    # print(x_val.shape)      # (24, 4)
    # print(x_test.shape)     # (30, 4)

for algorithm in model :
    score = cross_val_score(algorithm, x_train, y_train, cv=kfold)
    print('score : ', score, '-'+str(algorithm))

# score :  [1.         0.89473684 0.89473684 1.         1.        ] -LinearSVC()
# score :  [0.95       1.         0.94736842 0.89473684 1.        ] -SVC()
# score :  [1.         0.89473684 0.94736842 1.         1.        ] -KNeighborsClassifier()
# score :  [0.95       1.         0.94736842 0.94736842 0.94736842] -LogisticRegression()
# score :  [1.         0.89473684 1.         1.         0.94736842] -RandomForestClassifier()
# score :  [1.         1.         0.89473684 0.94736842 0.94736842] -DecisionTreeClassifier()