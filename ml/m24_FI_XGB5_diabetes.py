# xgboosting
# n_jobs 시간 확인
# >> -1 보다 코어 수를 정확하게 기재하는 게 더 빠를 수도 있다. >> 이것도 파라미터튜닝을 할 수 있다.


from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import pandas as pd
import numpy as np
import datetime

#1. DATA
dataset = load_diabetes()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['s4', 'sex', 's1'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

start = datetime.datetime.now()

#2. modeling
# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor(n_jobs=1)

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature_importances : \n", model.feature_importances_)  
print("acc : ", acc)  

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 에 따른 시간 차이 비교
# n_jobs=-1 time :  0:00:00.128655
# n_jobs=8  time :  0:00:00.128656 *
# n_jobs=4  time :  0:00:00.134639
# n_jobs=1  time :  0:00:00.140321


def cut_columns(feature_importances, columns, number):
    temp = []
    for i in feature_importances:
        temp.append(i)
    temp.sort()
    temp=temp[:number]
    result = []
    for j in temp:
        index = feature_importances.tolist().index(j)
        result.append(columns[index])
    return result

# print(cut_columns(model.feature_importances_, dataset.feature_names, 3))
# ['sex', 'age', 's1']


'''
# Graph : 컬럼 중 어떤 것이 가장 중요한 것인지 보여준다.
# 중요도가 낮은 컬럼은 제거해도 된다. >> 그만큼 자원이 절약된다.
import matplotlib.pyplot as plt
import numpy as np 

def plot_feature_importances_dataset(model) :
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
        align = 'center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)    # 축의 한계를 설정한다.

plot_feature_importances_dataset(model)
plt.show()
'''

# # DecisionTreeRegressor
# feature_importances : 
#  [0.02991191 0.         0.32054901 0.         0.01831924 0.06062798
#  0.         0.         0.57059185 0.        ]
# acc :  0.31490122539834386

# feature_importances : 
#  [0.04789328 0.3230994  0.04316533 0.58584199]
# acc :  0.3002403937235434

# # RandomForestRegressor
# feature_importances : 
#  [0.07030189 0.01003086 0.2502562  0.07685399 0.0498766  0.06264253
#  0.05161761 0.02060058 0.33629239 0.07152735]
# acc :  0.4211587112783205

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.08028076 0.26027649 0.09253146 0.09173989 0.06375254 0.33464406
#  0.07677481]
# acc :  0.375718550580461

# # GradientBoostingRegressor
# feature_importances : 
#  [0.07566439 0.01276838 0.27269052 0.08107363 0.03451144 0.06915807
#  0.04013073 0.01211572 0.35084826 0.05103885]
# acc :  0.36613284759593123

# 중요도 하위 25% 컬럼 제거
# feature_importances : 
#  [0.0685843  0.28607192 0.08046595 0.0897254  0.04175887 0.37419899
#  0.05919456]
# acc :  0.3899885195565056

# # xgboosting
# feature_importances : 
#  [0.0368821  0.03527097 0.15251055 0.05477958 0.04415327 0.06812558
#  0.0651588  0.05049536 0.42164674 0.0709771 ]
# acc :  0.24138193114785134

# 중요도 하위 25% 제거
# feature_importances : 
#  [0.04610109 0.17556281 0.06981704 0.09149326 0.06869822 0.45614192
#  0.09218565]
# acc :  0.20359464905750557

