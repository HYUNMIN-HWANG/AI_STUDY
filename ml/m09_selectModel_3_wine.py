# 모든 분류모델

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='classifier')    # type_filter='classifier' : 분류형 모델 전체를 불러온다.

for (name, algorithm) in allAlgorithms :    # 분류형 모델 전체를 반복해서 돌린다.
    # try ... except... : 예외처리 구문
    try :   # 에러가 없으면 아래 진행됨
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', accuracy_score(y_test, y_pred))
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# import sklearn
# print(sklearn.__version__)  # 0.23.2

'''
AdaBoostClassifier 의 정답률 :  0.8888888888888888
BaggingClassifier 의 정답률 :  0.9166666666666666
BernoulliNB 의 정답률 :  0.4166666666666667
CalibratedClassifierCV 의 정답률 :  0.9444444444444444
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 :  0.3888888888888889
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.6388888888888888
DecisionTreeClassifier 의 정답률 :  0.8611111111111112
DummyClassifier 의 정답률 :  0.16666666666666666
ExtraTreeClassifier 의 정답률 :  0.8888888888888888
ExtraTreesClassifier 의 정답률 :  0.9722222222222222
GaussianNB 의 정답률 :  0.9166666666666666
GaussianProcessClassifier 의 정답률 :  0.3888888888888889
GradientBoostingClassifier 의 정답률 :  0.9166666666666666
HistGradientBoostingClassifier 의 정답률 :  0.9444444444444444
KNeighborsClassifier 의 정답률 :  0.6944444444444444
LabelPropagation 의 정답률 :  0.5833333333333334
LabelSpreading 의 정답률 :  0.5833333333333334
LinearDiscriminantAnalysis 의 정답률 :  0.9722222222222222
LinearSVC 의 정답률 :  0.7777777777777778
LogisticRegression 의 정답률 :  0.9444444444444444
LogisticRegressionCV 의 정답률 :  0.8888888888888888
MLPClassifier 의 정답률 :  0.4722222222222222
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.7777777777777778
NearestCentroid 의 정답률 :  0.6388888888888888
NuSVC 의 정답률 :  0.8611111111111112
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.3055555555555556
Perceptron 의 정답률 :  0.6111111111111112
QuadraticDiscriminantAnalysis 의 정답률 :  1.0  ***
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9722222222222222
RidgeClassifier 의 정답률 :  0.9444444444444444
RidgeClassifierCV 의 정답률 :  0.9444444444444444
SGDClassifier 의 정답률 :  0.6111111111111112
SVC 의 정답률 :  0.6111111111111112
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''

# tensorflow _ dense
# acc : 1.0     ***