# gridSearch 단점 : 너무 느리다. 파라미터 100프로 모두 돌린다. 내가 지정한 파라미터를 100프로 신뢰할 수 없다.
# >> randomSearch : 
# >> RandomizedSearchCV : 모든 파라미터를 건드릴 필요가 없다. 랜덤하게 일부만 확인한다. 속도가 빠르다.

import numpy as np
from sklearn.datasets import load_iris

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

# 모델마다 나오는 결과 값을 비교한다.
# from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier  # Classifier : 분류모델
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression # 회귀가 아닌 분류 모델임

import warnings
warnings.filterwarnings('ignore')

import datetime

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
parameters=[
    {'n_estimators' : [100, 200, 300], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12, 14], 'min_samples_leaf' : [3, 7, 10], 'min_samples_split' : [2, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 9, 10], 'n_jobs' : [-1, 2, 4]},
    {'n_jobs' : [-1, 0, 1]}
]

#2. Modeling 
# model = SVC()
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold)
# 모델 : RandomForestClassifier()
# parameters : SVC에 들어가 있는 파라미터 값들 (딕셔너리 형태)
# cv=kfold : 5번 돌리겠다.
# 총 90번 모델이 돌아감


#3. Compile, Train
start = datetime.datetime.now()
model.fit(x_train, y_train)
end = datetime.datetime.now()
print("time :", end-start)   # time : 0:00:08.714285

#4. Evaluate, Predict
print("최적의 매개변수 : ", model.best_estimator_)
#  model.best_estimator_ : GridSearchCV에서 90번 돌린 것 중에서 어떤 파라미터가 가장 좋은 값인지 알려준다.

y_pred = model.predict(x_test)
print('최종정답률', accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)
print('aaa ', aaa)

# gridSearch
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, min_samples_leaf=7)
# 최종정답률 0.9333333333333333
# aaa  0.9333333333333333

# RandomSearch
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=9, n_jobs=-1)
# 최종정답률 0.9666666666666667
# aaa  0.9666666666666667