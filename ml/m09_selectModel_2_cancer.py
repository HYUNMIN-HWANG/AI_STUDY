# 모든 분류모델

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 :  0.9736842105263158
BaggingClassifier 의 정답률 :  0.9824561403508771
BernoulliNB 의 정답률 :  0.6578947368421053      
CalibratedClassifierCV 의 정답률 :  0.9824561403508771
CategoricalNB 은 없는 모델
CheckingClassifier 의 정답률 :  0.34210526315789475   
ClassifierChain 은 없는 모델
ComplementNB 의 정답률 :  0.9473684210526315
DecisionTreeClassifier 의 정답률 :  0.9385964912280702
DummyClassifier 의 정답률 :  0.5263157894736842       
ExtraTreeClassifier 의 정답률 :  0.9298245614035088   
ExtraTreesClassifier 의 정답률 :  0.9736842105263158
GaussianNB 의 정답률 :  0.9736842105263158
GaussianProcessClassifier 의 정답률 :  0.9298245614035088
GradientBoostingClassifier 의 정답률 :  0.9736842105263158
HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
KNeighborsClassifier 의 정답률 :  0.956140350877193
LabelPropagation 의 정답률 :  0.3684210526315789
LabelSpreading 의 정답률 :  0.3684210526315789
LinearDiscriminantAnalysis 의 정답률 :  0.9912280701754386  ***
LinearSVC 의 정답률 :  0.9298245614035088
LogisticRegression 의 정답률 :  0.9736842105263158
LogisticRegressionCV 의 정답률 :  0.9736842105263158
MLPClassifier 의 정답률 :  0.956140350877193
MultiOutputClassifier 은 없는 모델
MultinomialNB 의 정답률 :  0.9473684210526315
NearestCentroid 의 정답률 :  0.9298245614035088
NuSVC 의 정답률 :  0.9385964912280702
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.9473684210526315
Perceptron 의 정답률 :  0.8421052631578947
QuadraticDiscriminantAnalysis 의 정답률 :  0.9649122807017544
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9649122807017544
RidgeClassifier 의 정답률 :  0.9824561403508771
RidgeClassifierCV 의 정답률 :  0.9824561403508771
SGDClassifier 의 정답률 :  0.868421052631579
SVC 의 정답률 :  0.956140350877193
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''

# tensorflow _ CNN
# acc : 0.9824561476707458 