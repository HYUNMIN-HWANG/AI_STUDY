# 전처리 ; KFold
# 교차 검증값 : cross_val_score

import numpy as np
from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Regressor : 회귀모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

########################################################

#1. DATA
dataset = load_boston()
x = dataset.data 
y = dataset.target 

print(x.shape)  #(506, 13)
print(y.shape)  #(506,)

# train test split
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, random_state=77, shuffle=True, train_size=0.8)

# x preprocessing >>  K-Fold 
kfold = KFold(n_splits=5, shuffle=True) # 데이터를 5등분한다. > train data 와 validation data 로 구분한다.

# 머신러닝은 원핫인코딩 안해도 된다.

# 2. Modeling (한 줄로 끝)
models = [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]

for algorithm in models :  
    model = algorithm()
    scores = cross_val_score(model, x_train, y_train, cv=kfold) # r2_score
    print(algorithm)
    print('scores : ', scores)  


# <class 'sklearn.linear_model._base.LinearRegression'>
# scores :  [0.66031311 0.79113109 0.6974721  0.67542568 0.50987125]
# <class 'sklearn.neighbors._regression.KNeighborsRegressor'>
# scores :  [0.56847038 0.48594966 0.47877108 0.48731683 0.47390694]
# <class 'sklearn.tree._classes.DecisionTreeRegressor'>
# scores :  [0.45575103 0.60266924 0.80492126 0.75870768 0.73971235]
# <class 'sklearn.ensemble._forest.RandomForestRegressor'>
# scores :  [0.84630818 0.77729597 0.87321695 0.70826392 0.90749726]

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