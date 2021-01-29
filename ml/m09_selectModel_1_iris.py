# 모든 분류모델 all_estimators(type_filter='classifier') 

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_iris
import warnings

warnings.filterwarnings('ignore')

dataset = load_iris()
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
AdaBoostClassifier 의 정답률 :  0.9666666666666667
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9333333333333333
CategoricalNB 의 정답률 :  0.9
CheckingClassifier 의 정답률 :  0.3
TypeError: __init__() missing 1 required positional argument: 'base_estimator' 

>> 버전이 안 맞아서 생긴 오류 >> 예외처리를 만든다. (try...except...)

AdaBoostClassifier 의 정답률 :  0.9666666666666667
BaggingClassifier 의 정답률 :  0.9666666666666667
BernoulliNB 의 정답률 :  0.3
CalibratedClassifierCV 의 정답률 :  0.9333333333333333
CategoricalNB 의 정답률 :  0.9
CheckingClassifier 의 정답률 :  0.3
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.7
DecisionTreeClassifier 의 정답률 :  0.8666666666666667
DummyClassifier 의 정답률 :  0.36666666666666664
ExtraTreeClassifier 의 정답률 :  0.9
ExtraTreesClassifier 의 정답률 :  0.9666666666666667
GaussianNB 의 정답률 :  0.9333333333333333
GaussianProcessClassifier 의 정답률 :  0.9666666666666667
GradientBoostingClassifier 의 정답률 :  0.9666666666666667
HistGradientBoostingClassifier 의 정답률 :  0.9666666666666667
KNeighborsClassifier 의 정답률 :  0.9666666666666667
LabelPropagation 의 정답률 :  0.9666666666666667
LabelSpreading 의 정답률 :  0.9666666666666667
LinearDiscriminantAnalysis 의 정답률 :  1.0 ***
LinearSVC 의 정답률 :  0.9666666666666667
LogisticRegression 의 정답률 :  0.9666666666666667
LogisticRegressionCV 의 정답률 :  0.9666666666666667
MLPClassifier 의 정답률 :  1.0  ***
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.8666666666666667
NearestCentroid 의 정답률 :  0.9
NuSVC 의 정답률 :  0.9666666666666667
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.7
Perceptron 의 정답률 :  0.7333333333333333
QuadraticDiscriminantAnalysis 의 정답률 :  1.0  ***
RadiusNeighborsClassifier 의 정답률 :  0.9333333333333333
RandomForestClassifier 의 정답률 :  0.9333333333333333
RidgeClassifier 의 정답률 :  0.8333333333333334
RidgeClassifierCV 의 정답률 :  0.8333333333333334
SGDClassifier 의 정답률 :  0.7
SVC 의 정답률 :  0.9666666666666667
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''

# tensorflow _ CNN
# acc : 1.0 ***