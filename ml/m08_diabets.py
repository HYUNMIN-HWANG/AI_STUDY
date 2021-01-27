# 머신러닝 모델링 (회귀모델)
# model.score와 r2_score 동일하다


import numpy as np
from sklearn.datasets import load_diabetes

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
dataset = load_diabetes()
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
# model = LinearRegression()
# model = KNeighborsRegressor()
model = DecisionTreeRegressor()
# model = RandomForestRegressor()

#3. Compile, Train
# 머신러닝 단 한 줄!!!
model.fit(x_train, y_train)

#4. Evaluate, Predict
y_pred = model.predict(x_test)      
print("y_pred : ", y_pred)  

result = model.score(x_test, y_test)    # evaluate
print("model.score : ", result)         

r2 = r2_score(y_test, y_pred)
print("r2_score : ", r2)      


# model = LinearRegression()
# model.score :  0.5063891053505036
# r2_score :  0.5063891053505036 

# model = KNeighborsRegressor()
# model.score :  0.3741821819765594
# r2_score :  0.3741821819765594

# model = DecisionTreeRegressor()
# model.score :  -0.1966927464630237
# r2_score :  -0.1966927464630237

# model = RandomForestRegressor()
# model.score :  0.38175456606521874
# r2_score :  0.38175456606521874

# tensorflow _ dense (modelcheckpoint)
# r2 : 0.5510438539208138   ***