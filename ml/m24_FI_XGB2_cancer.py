# xgboosting
# n_jobs 시간 확인
# >> -1 보다 코어 수를 정확하게 기재하는 게 더 빠를 수도 있다. >> 이것도 파라미터튜닝을 할 수 있다.


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import datetime

import numpy as np
import pandas as pd

#1. DATA
dataset = load_breast_cancer()
x = dataset.data 
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names']) 
x = x_pd.drop(['worst compactness', 'mean symmetry', 'concave points error', 'mean perimeter', 'symmetry error', 'mean compactness', 'worst symmetry', 'mean fractal dimension'], axis=1)
x = x.to_numpy()

# print(x.shape)
# print(y.shape)


x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=44)

start = datetime.datetime.now()

#2. modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier(n_jobs = -1, use_label_encoder=False)      # n_jobs = -1 : CPU 자원을 모두 사용하겠다.
# model = XGBClassifier(n_jobs = 1)      

#3. Train
model.fit(x_train, y_train, eval_metric='logloss')

#4. Score, Predict
acc = model.score(x_test, y_test)

print("feature importances : \n", model.feature_importances_)  
print("acc : ", acc) 

end = datetime.datetime.now()
print("time : ", end-start)

# n_jobs 에 따른 시간 차이 비교
# n_jobs = -1 time :  0:00:00.074800
# n_jobs = 8  time :  0:00:00.069813 *
# n_jobs = 4  time :  0:00:00.086740
# n_jobs = 1  time :  0:00:00.101728


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
# ['worst compactness', 'mean symmetry', 'concave points error', 'mean perimeter', 'symmetry error', 'mean compactness', 'worst symmetry', 'mean fractal dimension']
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

## GradientBoostingClassifier()
# feature importances : 
#  [9.31935547e-07 1.30036623e-02 7.78317824e-04 5.54463596e-04
#  1.19753997e-03 2.73387337e-03 1.33698340e-03 4.08467469e-01
#  8.10046980e-05 2.64180475e-04 1.89295215e-03 1.82571810e-03
#  3.21419079e-04 7.24135593e-03 3.72634912e-04 1.08171438e-03
#  6.47708606e-04 1.02875138e-03 8.93329927e-04 3.58982200e-03
#  7.87588731e-02 5.86893598e-02 2.78478694e-01 4.58131841e-02
#  1.56396058e-02 5.49252681e-04 2.05173188e-02 5.19268774e-02
#  3.08287100e-04 2.00471411e-03]
# acc :  0.9824561403508771

# # 가장 작은 중요도 25% 삭제하라
# feature importances : 
#  [7.06911552e-03 2.67119217e-04 1.13691643e-03 3.73346979e-03
#  1.34515242e-03 4.06081632e-01 4.77823494e-03 2.70675093e-03
#  8.10866196e-03 6.05141258e-04 9.40066857e-04 1.52920498e-03
#  1.76484878e-03 6.70352976e-02 6.40680058e-02 2.85528523e-01
#  5.13873422e-02 1.38520002e-02 5.75095695e-04 1.90737017e-02
#  5.52623398e-02 3.15137840e-03]
# acc :  0.9649122807017544

# # xgboosting
# feature importances : 
#  [4.9588699e-03 1.5423428e-02 6.3683779e-04 2.2023508e-02 7.9919444e-03
#  9.5059077e-04 1.1493003e-02 1.5286781e-01 3.9727203e-04 2.7880725e-03
#  3.4555981e-03 5.6501115e-03 2.8066258e-03 3.8451280e-03 2.8832396e-03
#  3.3951809e-03 3.4120989e-03 5.8727746e-04 9.2994876e-04 3.7213673e-03
#  1.1663227e-01 2.0733794e-02 5.1057857e-01 1.3440680e-02 6.8047098e-03
#  0.0000000e+00 2.1567145e-02 5.5507991e-02 1.3584920e-03 3.1585069e-03]
# acc :  0.9824561403508771

# 중요도 하위 25% 제거
# feature importances :
#  [0.00555774 0.01792266 0.01265311 0.00740111 0.009242   0.17113215
#  0.00407904 0.00319243 0.00579982 0.00433497 0.00251646 0.00312372
#  0.00087511 0.00360278 0.1307501  0.02021912 0.49249628 0.01581706
#  0.00712727 0.01991527 0.05881077 0.00343105]
# acc :  0.9736842105263158