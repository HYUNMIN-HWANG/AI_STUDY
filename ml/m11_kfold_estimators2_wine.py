# 모든 분류모델 kfold 적용

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')    # type_filter='classifier' : 분류형 모델 전체를 불러온다.

for (name, algorithm) in allAlgorithms :    # 분류형 모델 전체를 반복해서 돌린다.
    # try ... except... : 예외처리 구문
    try :   # 에러가 없으면 아래 진행됨
        model = algorithm()
        score = cross_val_score(model, x_train, y_train, cv=kfold)
        # model.fit(x_train, y_train)
        print(name, '의 정답률 : \n', score)
        # y_pred = model.predict(x_test)
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# import sklearn
# print(sklearn.__version__)  # 0.23.2

'''
AdaBoostClassifier 의 정답률 : 
 [0.96551724 0.93103448 0.89285714 0.82142857 0.35714286]
BaggingClassifier 의 정답률 : 
 [0.96551724 0.82758621 1.         0.96428571 0.96428571]
BernoulliNB 의 정답률 :
 [0.4137931  0.4137931  0.53571429 0.28571429 0.32142857]
CalibratedClassifierCV 의 정답률 : 
 [0.86206897 0.89655172 0.92857143 0.92857143 1.        ]
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 : 
 [0.68965517 0.62068966 0.53571429 0.75       0.71428571]
DecisionTreeClassifier 의 정답률 :
 [0.89655172 0.93103448 0.89285714 0.92857143 0.82142857]
DummyClassifier 의 정답률 :
 [0.31034483 0.27586207 0.5        0.5        0.25      ]
ExtraTreeClassifier 의 정답률 :
 [0.96551724 0.86206897 0.92857143 0.78571429 0.89285714]
ExtraTreesClassifier 의 정답률 : 
 [1.         1.         1.         1.         0.96428571]
GaussianNB 의 정답률 :
 [1.         0.93103448 1.         1.         0.96428571]
GaussianProcessClassifier 의 정답률 : 
 [0.51724138 0.51724138 0.46428571 0.42857143 0.5       ]
GradientBoostingClassifier 의 정답률 : 
 [0.93103448 0.86206897 0.89285714 0.96428571 1.        ]
HistGradientBoostingClassifier 의 정답률 : 
 [1.         0.96551724 1.         0.89285714 1.        ]
KNeighborsClassifier 의 정답률 :
 [0.75862069 0.68965517 0.67857143 0.67857143 0.85714286]
LabelPropagation 의 정답률 :
 [0.4137931  0.4137931  0.57142857 0.42857143 0.39285714]
LabelSpreading 의 정답률 :
 [0.48275862 0.31034483 0.53571429 0.5        0.39285714]
LinearDiscriminantAnalysis 의 정답률 : 
 [1.         0.96551724 1.         1.         0.92857143]
LinearSVC 의 정답률 : 
 [0.65517241 0.68965517 0.85714286 0.85714286 0.92857143]
LogisticRegression 의 정답률 : 
 [0.89655172 0.93103448 0.96428571 0.96428571 0.85714286]
LogisticRegressionCV 의 정답률 : 
 [0.96551724 0.93103448 0.92857143 0.96428571 0.96428571]
MLPClassifier 의 정답률 : 
 [0.44827586 0.86206897 0.57142857 0.25       0.07142857]
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :
 [0.72413793 0.75862069 0.75       0.89285714 1.        ]
NearestCentroid 의 정답률 :
 [0.82758621 0.75862069 0.71428571 0.75       0.71428571]
NuSVC 의 정답률 :
 [1.         0.86206897 0.85714286 0.78571429 0.78571429]
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :
 [0.55172414 0.55172414 0.5        0.78571429 0.64285714]
Perceptron 의 정답률 : 
 [0.44827586 0.48275862 0.60714286 0.32142857 0.53571429]
QuadraticDiscriminantAnalysis 의 정답률 :
 [0.96551724 0.93103448 1.         1.         1.        ]
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 : 
 [1.         1.         0.96428571 0.96428571 1.        ]
RidgeClassifier 의 정답률 :
 [1.         0.93103448 0.96428571 1.         0.96428571]
RidgeClassifierCV 의 정답률 : 
 [0.96551724 1.         0.96428571 0.96428571 1.        ]
SGDClassifier 의 정답률 :
 [0.68965517 0.62068966 0.5        0.67857143 0.46428571]
SVC 의 정답률 : 
 [0.68965517 0.65517241 0.67857143 0.67857143 0.71428571]
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''

# tensorflow _ dense
# acc : 1.0     ***