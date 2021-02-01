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
from xgboost import XGBRegressor

#1. DATA
datasets = load_iris()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)
# print(x.shape[1])

for i in range(x.shape[1]) : 
    i = i + 1
    pca = PCA(n_components=i)
    x2 = pca.fit_transform(x)  # fit_transform : 전처리 fit과 transform 한꺼번에 한다.

    x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=46)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # print(x_train.shape)            # (120, 2) >> 컬럼을 압축시켰다. 컬럼 재구성됨
    # print(x_test.shape)             # (30, 2) >> 컬럼을 압축시켰다. 컬럼 재구성됨

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
    model = RandomForestClassifier()
    # model = XGBRegressor(n_jobs=-1, use_label_encoder=False)

    #3. Train
    model.fit(x_train, y_train)
    # model.fit(x_train, y_train, eval_metric='logloss')

    #4. Score, Predict
    print("n_components ", i)
    result = model.score(x_test, y_test)
    print("model.score : ", result)

    y_pred = model.predict(x_test)

    score = accuracy_score(y_pred, y_test)
    print("accuracy_score : ", score)

    # pca / RandomForest
    # n_components  1
    # model.score :  0.9
    # r2_score :  0.845890410958904
    # n_components  2
    # model.score :  0.9333333333333333
    # r2_score :  0.9033816425120773
    # n_components  3
    # model.score :  0.9
    # r2_score :  0.8628048780487805
    # n_components  4
    # model.score :  0.9333333333333333
    # r2_score :  0.9033816425120773

    # XGBoost
    # n_components  1
    # model.score :  0.8555486561667479
    # r2_score :  0.846368564551409
    # n_components  2
    # model.score :  0.9235155731285878
    # r2_score :  0.9211691302103862
    # n_components  3
    # model.score :  0.9026720046667223
    # r2_score :  0.9002417209742706
    # n_components  4
    # model.score :  0.884639105211108
    # r2_score :  0.8775476678201166