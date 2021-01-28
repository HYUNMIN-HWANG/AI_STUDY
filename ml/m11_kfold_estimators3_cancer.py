# 모든 분류모델 kfold 적용

# 모든 분류모델

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
        # y_pred = model.predict(x_test)
        print(name, '의 정답률 : \n', score)
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# import sklearn
# print(sklearn.__version__)  # 0.23.2

'''
AdaBoostClassifier 의 정답률 : 
 [0.97802198 0.95604396 0.92307692 0.98901099 0.93406593]
BaggingClassifier 의 정답률 : 
 [0.98901099 0.95604396 0.89010989 0.94505495 0.91208791]
BernoulliNB 의 정답률 :
 [0.56043956 0.63736264 0.54945055 0.71428571 0.63736264]
CalibratedClassifierCV 의 정답률 : 
 [0.9010989  0.93406593 0.94505495 0.92307692 0.9010989 ]
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 :
 [0. 0. 0. 0. 0.]
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 : 
 [0.86813187 0.9010989  0.86813187 0.85714286 0.91208791]
DecisionTreeClassifier 의 정답률 : 
 [0.91208791 0.93406593 0.94505495 0.85714286 0.95604396]
DummyClassifier 의 정답률 :
 [0.56043956 0.45054945 0.63736264 0.48351648 0.57142857]
ExtraTreeClassifier 의 정답률 : 
 [0.87912088 0.9010989  0.9010989  0.91208791 0.91208791]
ExtraTreesClassifier 의 정답률 : 
 [0.95604396 0.97802198 0.98901099 0.94505495 0.98901099]
GaussianNB 의 정답률 :
 [0.96703297 0.92307692 0.95604396 0.87912088 0.92307692]
GaussianProcessClassifier 의 정답률 : 
 [0.9010989  0.87912088 0.91208791 0.91208791 0.9010989 ]
GradientBoostingClassifier 의 정답률 : 
 [0.95604396 0.94505495 0.94505495 0.94505495 0.94505495]
HistGradientBoostingClassifier 의 정답률 : 
 [0.97802198 0.98901099 0.94505495 0.93406593 0.95604396]
KNeighborsClassifier 의 정답률 : 
 [0.93406593 0.91208791 0.93406593 0.9010989  0.92307692]
LabelPropagation 의 정답률 : 
 [0.42857143 0.40659341 0.37362637 0.36263736 0.41758242]
LabelSpreading 의 정답률 : 
 [0.36263736 0.43956044 0.36263736 0.41758242 0.38461538]
LinearDiscriminantAnalysis 의 정답률 :
 [0.95604396 0.96703297 0.94505495 0.94505495 0.95604396]
LinearSVC 의 정답률 : 
 [0.93406593 0.92307692 0.9010989  0.9010989  0.56043956]
LogisticRegression 의 정답률 : 
 [0.94505495 0.9010989  0.94505495 0.97802198 0.92307692]
LogisticRegressionCV 의 정답률 : 
 [0.96703297 0.96703297 0.95604396 0.96703297 0.97802198]
MLPClassifier 의 정답률 : 
 [0.94505495 0.92307692 0.9010989  0.93406593 0.95604396]
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :
 [0.86813187 0.91208791 0.91208791 0.85714286 0.84615385]
NearestCentroid 의 정답률 : 
 [0.93406593 0.9010989  0.89010989 0.82417582 0.85714286]
NuSVC 의 정답률 : 
 [0.91208791 0.83516484 0.89010989 0.86813187 0.83516484]
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :
 [0.9010989  0.82417582 0.9010989  0.83516484 0.93406593]
Perceptron 의 정답률 : 
 [0.81318681 0.85714286 0.79120879 0.86813187 0.89010989]
QuadraticDiscriminantAnalysis 의 정답률 :
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :
 [0.93406593 0.95604396 0.95604396 0.97802198 0.96703297]
RidgeClassifier 의 정답률 :
 [0.94505495 0.96703297 0.92307692 0.95604396 0.94505495]
RidgeClassifierCV 의 정답률 :
 [0.98901099 0.93406593 0.92307692 0.96703297 0.97802198]
SGDClassifier 의 정답률 :
 [0.9010989  0.9010989  0.84615385 0.86813187 0.9010989 ]
SVC 의 정답률 :
 [0.89010989 0.93406593 0.91208791 0.9010989  0.91208791]
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''

# tensorflow _ CNN
# acc : 0.9824561476707458 