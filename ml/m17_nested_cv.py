# train / test / val 

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

import warnings
warnings.filterwarnings('ignore')

########################################################

#1. DATA
dataset = load_iris()
x = dataset.data 
y = dataset.target 

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )

# preprocessing >>  K-Fold 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 데이터를 5등분한다. > train data 와 test data 로 구분한다.

# dictionary 3개 (key-value 쌍) - SVC parameters에 해당하는 값들
parameters = [
    {"C" : [1, 10, 100, 1000], "kernel" : ["linear"]},                              # 4번 계산
    {"C" : [1, 10, 100], "kernel" : ["rbf"], "gamma" : [0.001, 0.0001]},            # 6번 계산
    {"C" : [1, 10, 100, 1000], "kernel" : ["sogmoid"], "gamma" : [0.001, 0.0001]}   # 8번 계산
]   # 한 번 kfold를 돌 때마다 총 18번 파라미터 계산함

#2. Modeling 

# 교차검증
model = GridSearchCV(SVC(), parameters, cv=kfold)           # 5번 돌아감
score = cross_val_score(model, x_train, y_train, cv=kfold)  # 5번 돌아감 ==> 총 25번 돌아간다.

print('교차검증점수 : ', score)
# 교차검증점수 :  [0.95833333 1.         0.91666667 0.95833333 0.95833333]


