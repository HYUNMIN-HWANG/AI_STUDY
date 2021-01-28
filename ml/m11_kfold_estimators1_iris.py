# 모든 분류모델 kfold 적용

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='classifier')    # type_filter='classifier' : 분류형 모델 전체를 불러온다.

for (name, algorithm) in allAlgorithms :    # 분류형 모델 전체를 반복해서 돌린다.
    # try ... except... : 예외처리 구문
    try :   # 에러가 없으면 아래 진행됨
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold)  # accuracy_score 가 다섯개씩 출력된다.
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 : \n', scores) # 5개 씩 score
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# import sklearn
# print(sklearn.__version__)  # 0.23.2

'''
 [0.625      0.875      1.         0.95833333 0.91666667]
BaggingClassifier 의 정답률 : 
 [0.91666667 0.95833333 1.         0.91666667 1.        ]
BernoulliNB 의 정답률 :
 [0.20833333 0.16666667 0.125      0.25       0.33333333]
CalibratedClassifierCV 의 정답률 : 
 [0.91666667 0.91666667 0.875      0.95833333 0.95833333]
CategoricalNB 의 정답률 :
 [0.875      0.875      1.         0.95833333 0.95833333]
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :
 [0.5        0.66666667 0.75       0.625      0.75      ]
DecisionTreeClassifier 의 정답률 : 
 [1.         0.91666667 0.91666667 0.95833333 0.95833333]
DummyClassifier 의 정답률 :
 [0.16666667 0.25       0.45833333 0.33333333 0.25      ]
ExtraTreeClassifier 의 정답률 : 
 [0.91666667 0.91666667 0.91666667 1.         0.95833333]
ExtraTreesClassifier 의 정답률 : 
 [0.95833333 0.91666667 0.91666667 0.95833333 1.        ]
GaussianNB 의 정답률 :
 [0.95833333 1.         0.95833333 0.95833333 0.91666667]
GaussianProcessClassifier 의 정답률 : 
 [0.91666667 1.         0.95833333 0.91666667 0.91666667]
GradientBoostingClassifier 의 정답률 : 
 [0.95833333 0.95833333 0.91666667 1.         0.91666667]
HistGradientBoostingClassifier 의 정답률 : 
 [0.91666667 0.95833333 0.91666667 0.95833333 0.95833333]
KNeighborsClassifier 의 정답률 :
 [0.95833333 0.95833333 1.         0.91666667 0.95833333]
LabelPropagation 의 정답률 :
 [0.95833333 0.95833333 1.         1.         0.875     ]
LabelSpreading 의 정답률 : 
 [0.875      1.         0.95833333 0.95833333 1.        ]
LinearDiscriminantAnalysis 의 정답률 :
 [0.95833333 1.         0.95833333 0.95833333 1.        ]
LinearSVC 의 정답률 : 
 [1.         0.91666667 0.91666667 0.91666667 1.        ]
LogisticRegression 의 정답률 : 
 [0.95833333 0.91666667 0.91666667 1.         1.        ]
LogisticRegressionCV 의 정답률 : 
 [1.         0.875      0.91666667 0.95833333 1.        ]
MLPClassifier 의 정답률 : 
 [0.95833333 0.91666667 0.95833333 1.         0.95833333]
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :
 [0.95833333 0.91666667 0.58333333 0.875      1.        ]
NearestCentroid 의 정답률 :
 [0.91666667 0.91666667 0.91666667 0.95833333 0.95833333]
NuSVC 의 정답률 : 
 [0.95833333 0.91666667 0.95833333 1.         0.875     ]
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :
 [0.83333333 0.83333333 0.70833333 0.95833333 0.875     ]
Perceptron 의 정답률 : 
 [0.45833333 0.75       0.70833333 0.70833333 0.375     ]
QuadraticDiscriminantAnalysis 의 정답률 :
 [1.         1.         0.91666667 1.         1.        ]
RadiusNeighborsClassifier 의 정답률 : 
 [0.91666667 0.95833333 1.         0.95833333 0.875     ]
RandomForestClassifier 의 정답률 : 
 [0.95833333 0.95833333 0.91666667 0.95833333 0.91666667]
RidgeClassifier 의 정답률 :
 [0.70833333 0.83333333 0.91666667 0.75       0.875     ]
RidgeClassifierCV 의 정답률 : 
 [0.83333333 0.79166667 0.83333333 1.         0.75      ]
SGDClassifier 의 정답률 : 
 [0.95833333 0.58333333 0.95833333 0.875      0.91666667]
SVC 의 정답률 :
 [0.95833333 1.         0.91666667 0.875      1.        ]
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''

# tensorflow _ CNN
# acc : 1.0 ***