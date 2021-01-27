# 머신러닝 모델링 (다중분류모델)
# model.score와 accuracy_score 동일하다


import numpy as np
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

###############################################

#1. DATA
dataset = load_wine()
x = dataset.data
y = dataset.target 

print(x.shape)  # (178, 13)
print(y.shape)  # (178,)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=0.8, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. Modeling
# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

#3. Train
model.fit(x_train, y_train)

#4. Evaluate, Predict
y_pred = model.predict(x_test)
print("y_pred : ", y_pred)

result = model.score(x_test, y_test)
print("model.score : ", result)

acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

# model = LinearSVC()
# y_pred :  [2 1 1 0 1 1 2 0 0 2 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222

# model = SVC()
# y_pred :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# model.score :  1.0
# accuracy_score :  1.0

# model = KNeighborsClassifier()
# y_pred :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# model.score :  1.0
# accuracy_score :  1.0

# model = DecisionTreeClassifier()
# y_pred :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 2 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# model.score :  0.9722222222222222
# accuracy_score :  0.9722222222222222

# model = RandomForestClassifier()
# y_pred :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# model.score :  1.0
# accuracy_score :  1.0

# model = LogisticRegression()
# y_pred :  [2 1 1 0 1 1 2 0 0 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 0 0 1 1 0 1 1 0 2 0 1 0]
# model.score :  1.0
# accuracy_score :  1.0

# tensorflow _ dense
# acc : 1.0     ***
