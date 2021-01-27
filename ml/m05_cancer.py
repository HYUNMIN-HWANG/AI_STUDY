# 머신러닝 모델링 (이진분류모델)
# model.score와 accuracy_score 동일하다


import numpy as np
from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Classifier : 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

########################################################


#1. DATA
dataset = load_breast_cancer()
x = dataset.data 
y = dataset.target 

print(x.shape)  # (442, 10)
print(y.shape)  # (442,)

# x값 전처리
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 머신러닝은 원핫인코딩 안해도 된다.

#2. Modeling (한 줄로 끝)
# model = LinearSVC()
# model = SVC()
model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = LogisticRegression()

#3. Compile, Train
# 머신러닝 단 한 줄!!!
model.fit(x_train, y_train)


#4. Evaluate, Predict

y_pred = model.predict(x_test)      
print("y_pred : ", y_pred)  

result = model.score(x_test, y_test)
print("model.score : ", result)         

acc = accuracy_score(y_test, y_pred)
print("accuract.score : ", acc)      

# model = LinearSVC()
# model.score :  0.9736842105263158
# accuract.score :  0.9736842105263158

# model = SVC()
# model.score :  0.9736842105263158
# accuract.score :  0.9736842105263158

# model = KNeighborsClassifier()
# model.score :  0.956140350877193
# accuract.score :  0.956140350877193

# model = DecisionTreeClassifier()
# model.score :  0.9473684210526315
# accuract.score :  0.9473684210526315

# model = RandomForestClassifier()
# model.score :  0.956140350877193
# accuract.score :  0.956140350877193

# model = LogisticRegression()
# model.score :  0.9649122807017544
# accuract.score :  0.9649122807017544

# tensorflow _ CNN
# acc : 0.9824561476707458  ***