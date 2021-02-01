# PCA : 차원축소, 컬럼 재구성
# RandomForest로 모델링

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

#1. DATA
datasets = load_wine()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (178, 13) (178,)

pca = PCA(n_components=2)
x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=46)

print(x_train.shape)            # (142, 2)   >> 컬럼을 압축시켰다. 컬럼 재구성됨
print(x_test.shape)             # (36, 2)    >> 컬럼을 압축시켰다. 컬럼 재구성됨


# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum)  # cumsum 누적 합을 계산
# cumsum :  [0.99809123 0.99982715 0.99992211 0.99997232 0.99998469 0.99999315
#  0.99999596 0.99999748 0.99999861 0.99999933 0.99999971 0.99999992
#  1.        ]

# d = np.argmax(cumsum >= 0.99)+1
# print("cumsum >= 0.99", cumsum > 0.99)
# print("d : ", d)
# cumsum >= 0.99 [ True  True  True  True  True  True  True  True  True  True  True  True  True]
# d :  1

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

#2. Modeling
# model = Pipeline([("scaler", MinMaxScaler()),("model",RandomForestRegressor())])
model = Pipeline([("scaler", MinMaxScaler()),("model",XGBClassifier())])

#3. Train
# model.fit(x_train, y_train)
model.fit(x_train, y_train)

#4. Score, Predict
result = model.score(x_test, y_test)
print("model.score : ", result)

y_pred = model.predict(x_test)

score = accuracy_score(y_pred, y_test)
print("accuracy_score : ", score)

# RandomForest
# model.score :  0.16912835051546382
# r2_score :  0.040802995015498

# XGboost
# model.score :  0.6944444444444444
# r2_score :  -0.01098901098901095