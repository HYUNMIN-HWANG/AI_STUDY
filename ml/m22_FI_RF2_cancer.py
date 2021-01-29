# feature_importances
# 중요도가 낮은 컬럼 제거한 후 실행 >> 제거하기 전이랑 결과 유사하다

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#1. DATA
dataset = load_breast_cancer()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['symmetry error', 'smoothness error', 'mean fractal dimension', 'fractal dimension error', 'mean symmetry', 'compactness error', 'perimeter error', 'concavity error'], axis=1)
x = x.to_numpy()

# print(x.shape)
# print(y.shape)


x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

#2. modeling
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature importances : \n", model.feature_importances_)  
print("acc : ", acc) 

# 중요도 낮은 피처
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

# print(cut_columns(model.feature_importances_, dataset.feature_names, 8))
# ['symmetry error', 'smoothness error', 'mean fractal dimension', 'fractal dimension error', 'mean symmetry', 'compactness error', 'perimeter error', 'concavity error'] 

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

## DecisionTreeClassifier
# feature importances : 
# [0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.         0.         0.
#  0.         0.         0.         0.05612587 0.78678449 0.01008994
#  0.02293065 0.         0.         0.12406904 0.         0.        ]
# acc :  0.9385964912280702

# 중요도 0인 컬럼 삭제한 후 >> 삭제하거나 그대로나 acc 유사하다
# feature importances :
#  [0.05612587 0.78000877 0.01686566 0.00995429 0.1370454 ]
# acc :  0.9385964912280702

## RandomForest
# feature importances : 
#  [0.03734963 0.01142124 0.05376611 0.04248085 0.00957115 0.01117671
#  0.04998963 0.14707321 0.0037625  0.00364357 0.00841149 0.00533809 
#  0.00956402 0.02601971 0.0045359  0.00477779 0.00683035 0.00603496 
#  0.00264008 0.00419666 0.09315835 0.01930668 0.14582315 0.10314911 
#  0.01624926 0.0151702  0.03800293 0.09964593 0.01145739 0.00945334]
# acc :  0.9649122807017544

# # 가장 작은 중요도 25% 삭제하라
# feature importances : 
#  [0.05920514 0.01985096 0.03669769 0.02640441 0.00916818 0.01776303
#  0.0645727  0.08562135 0.01033007 0.00656798 0.02017784 0.00615977
#  0.12374892 0.02101449 0.13629975 0.15210707 0.01786766 0.01279145
#  0.048276   0.10083957 0.01570647 0.00882951]
# acc :  0.9649122807017544