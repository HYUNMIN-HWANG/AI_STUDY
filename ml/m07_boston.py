# 머신러닝 모델링 (회귀모델)
# model.score와 r2_score 동일하다


import numpy as np
from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# 모델마다 나오는 결과 값을 비교한다.
# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Regressor : 회귀모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression   # 이거는 분류모델

########################################################

#1. DATA
dataset = load_boston()
x = dataset.data 
y = dataset.target 

print(x.shape)  # (506, 13)
print(y.shape)  # (506,)

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
# model = LinearRegression()
# model = KNeighborsRegressor()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()


#3. Compile, Train
# 머신러닝 단 한 줄!!!
model.fit(x_train, y_train)


#4. Evaluate, Predict

y_pred = model.predict(x_test)      
print("y_pred : ", y_pred)  

result = model.score(x_test, y_test)    # evaluate >> r2 score
print("model.score : ", result)         

r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)      

# model = LinearRegression()
# model.score :  0.8111288663608667
# r2_score :  0.8111288663608667

# model = KNeighborsRegressor()
# model.score :  0.8265307833211177
# r2_score :  0.8265307833211177

# model = DecisionTreeRegressor()
# model.score :  0.8073382461997793
# r2_score :  0.8073382461997793

# model = RandomForestRegressor()
# model.score :  0.9213371468697082
# r2_score :  0.9213371468697082

# tensorflow _ cnn
# r2 : 0.9304400277059781   ***
