# PCA : 차원축소, 컬럼 재구성
# RandomForest로 모델링

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

#1. DATA
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

pca = PCA(n_components=2)
x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=46)

print(x_train.shape)            # (120, 2) >> 컬럼을 압축시켰다. 컬럼 재구성됨
print(x_test.shape)             # (30, 2) >> 컬럼을 압축시켰다. 컬럼 재구성됨

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)   
# print("cumsum : ", cumsum)  # cumsum 누적 합을 계산
# cumsum :  [0.92461872 0.97768521 0.99478782 1.        ]

# d = np.argmax(cumsum >= 0.95)+1 # cumsum이 0.95 이상인 컬럼을 True 로 만든다.
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)
# cumsum >= 0.95 [False  True  True  True]
# d :  2

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()
# plt.show()

#2. Modeling
model = Pipeline([("scaler", MinMaxScaler()),("model",RandomForestRegressor())])

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
result = model.score(x_test, y_test)
print("model.score : ", result)

y_pred = model.predict(x_test)

score = r2_score(y_pred, y_test)
print("r2_score : ", score)

# pca
# model.score :  0.9007391304347826
# r2_score :  0.897534978254795
