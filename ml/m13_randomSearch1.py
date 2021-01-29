# gridSearch 단점 : 너무 느리다. 파라미터 100프로 모두 돌린다. 내가 지정한 파라미터를 100프로 신뢰할 수 없다.
# >> RandomizedSearchCV (랜덤서치): 모든 파라미터를 건드릴 필요가 없다. 랜덤하게 일부만 확인한다. 속도가 빠르다.
# >> 기본값 n_iter = 10 

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

import warnings
warnings.filterwarnings('ignore')

import pandas as pd 

########################################################

#1. DATA
# dataset = load_iris()
# x = dataset.data 
# y = dataset.target 

# print(x.shape)  #(150, 4)
# print(y.shape)  #(150, )

# csv로 불러오기
dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)

x = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]

# print(x.shape, y.shape) # (150, 4) (150,)

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
# model = SVC()
# model = GridSearchCV(SVC(), parameters, cv=kfold)
model = RandomizedSearchCV(SVC(), parameters, cv=kfold)
# 모델 : SVC 모델을 RandomizedSearchCV로 쌓아버리겠다.
# parameters : SVC에 들어가 있는 파라미터 값들 (딕셔너리 형태)
# cv=kfold : 5번 돌리겠다.
# 총 90번 모델이 돌아감


#3. Compile, Train
model.fit(x_train, y_train)

#4. Evaluate, Predict
print("최적의 매개변수 : ", model.best_estimator_)
#  model.best_estimator_ : GridSearchCV에서 90번 돌린 것 중에서 어떤 파라미터가 가장 좋은 값인지 알려준다.

y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print('aaa ', aaa)

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최종정답률 0.9666666666666667
# aaa  0.9666666666666667
