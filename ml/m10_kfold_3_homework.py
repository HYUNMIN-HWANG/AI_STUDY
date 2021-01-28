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

# 머신러닝은 원핫인코딩 안해도 된다.

#2. Modeling (한 줄로 끝)
model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

# train 과 test 부분을 5등분한다.
scores = cross_val_score(model, x, y, cv=kfold)
# print(x.shape)  # (150, 4)

# train test split
x_train, x_val, y_train, y_val = \
    train_test_split(x, y, random_state=77, shuffle=True, train_size=0.8)

print('scores : ', scores)  
# model.fit과 model.score 한꺼번에 실행됨
# 결과 : model.score에 해당하는 값이 5번 출력된다.
# train / test kfold >> scores :  [0.9        0.96666667 0.93333333 1.         1.        ]
# train / test split >> train / validation kfold >> scores :  [0.83333333 1.         1.         1.         0.91666667]
# train / test kfold >> train / validation split >> scores :  [0.93333333 0.93333333 1.         0.96666667 0.96666667]
# print(x_train.shape)    # (120, 4)
# print(x_val.shape)      # (30, 4)

"""
#3. Compile, Train

model.fit(x_train, y_train)


#4. Evaluate, Predict

y_pred = model.predict(x_test)      
print("y_pred : ", y_pred)  

result = model.score(x_test, y_test)    # evaluate
print("model.score : ", result)         

acc = accuracy_score(y_test, y_pred)    # acc
print("accuracy_score : ", acc)      

# model = LinearSVC()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = SVC()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# model.score :  1.0
# accuracy_score :  1.0

# model = KNeighborsClassifier()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# model.score :  1.0
# accuracy_score :  1.0

# model = DecisionTreeClassifier()
# y_pred :  [1 1 1 0 1 1 0 0 0 1 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = RandomForestClassifier()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = LogisticRegression()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# tensorflow _ CNN
# acc : 1.0 ***
"""