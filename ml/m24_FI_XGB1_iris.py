# xgboosting (pip install xgboost)
# n_jobs 시간 확인 
# >> -1 보다 코어 수를 정확하게 기재하는 게 더 빠를 수도 있다. >> 이것도 파라미터튜닝을 할 수 있다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd
import numpy as np 
import datetime

#1. DATA
dataset = load_iris()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['sepal width (cm)'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

start = datetime.datetime.now()

#2. modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)      # n_jobs = -1 : CPU 자원을 모두 사용하겠다.

#3. Train
model.fit(x_train, y_train, eval_metric='logloss')

#4. Score, Predict
acc = model.score(x_test, y_test)

print(model.feature_importances_)  
print("acc : ", acc)  

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 에 따른 시간 차이 비교
# n_jobs=-1 time :  0:00:00.086739 
# n_jobs=8  time :  0:00:00.095744 
# n_jobs=1  time :  0:00:00.077831 *
# n_jobs=4  time :  0:00:00.091923


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

# print(cut_columns(model.feature_importances_, dataset.feature_names, 1)) # ['sepal width (cm)']
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

# # DecisionTreeClassifier
# feature_importances : [0.00787229 0.         0.4305627  0.56156501]
# acc :  0.9333333333333333

# 중요도 0인 값 제거
# [0.4305627 0.5694373]
# acc :  0.9

# # RandomForestClassifier
# [0.11326042 0.02297823 0.44464351 0.41911784]
# acc :  0.9666666666666667

# 중요도 하위 25% 제거
# [0.21338017 0.3748333  0.41178653]
# acc :  0.9333333333333333

# # GradientBoostingClassifier()
# [0.00718567 0.01234878 0.66303558 0.31742997]
# acc :  0.9666666666666667

# 중요도 하위 25% 제거
# [0.01603461 0.60020388 0.38376152]
# acc :  0.9666666666666667

# # xgboosting
# [0.02323038 0.01225644 0.8361378  0.12837538]
# acc :  0.9666666666666667

# 중요도 하위 25% 제거
# [0.02233907 0.86344945 0.11421151]
# acc :  0.9666666666666667
