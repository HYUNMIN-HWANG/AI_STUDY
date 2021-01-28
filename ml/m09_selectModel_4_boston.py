# 모든 회귀모델

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter='regressor')    # type_filter='regressor' : 회귀형 모델 전체를 불러온다.

for (name, algorithm) in allAlgorithms :    # 분류형 모델 전체를 반복해서 돌린다.
    # try ... except... : 예외처리 구문
    try :   # 에러가 없으면 아래 진행됨
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except :          #에러가 발생하면
        # continue    # 정지시키지 않고 계속 진행시키겠다.
        print(name, "은 없는 모델") # 예외처리한 모델 이름을 출력 

# import sklearn
# print(sklearn.__version__)  # 0.23.2

'''
ARDRegression 의 정답률 :  0.7512651671065581
AdaBoostRegressor 의 정답률 :  0.8402098233172509
BaggingRegressor 의 정답률 :  0.8905058919157057
BayesianRidge 의 정답률 :  0.7444785336818114
CCA 의 정답률 :  0.7270542664211517
DecisionTreeRegressor 의 정답률 :  0.8344857598894202
DummyRegressor 의 정답률 :  -0.0007982049217318821
ElasticNet 의 정답률 :  0.6990500898755508
ElasticNetCV 의 정답률 :  0.6902681369495264
ExtraTreeRegressor 의 정답률 :  0.7301482533993258
ExtraTreesRegressor 의 정답률 :  0.9030012741136 ***
GammaRegressor 의 정답률 :  -0.0007982049217318821
GaussianProcessRegressor 의 정답률 :  -5.639147690233129
GeneralizedLinearRegressor 의 정답률 :  0.6917874063129013
GradientBoostingRegressor 의 정답률 :  0.8950567993580895
HistGradientBoostingRegressor 의 정답률 :  0.8991491407747458
HuberRegressor 의 정답률 :  0.7233379135400204
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 정답률 :  0.6390759816821279
KernelRidge 의 정답률 :  0.7744886782300767
Lars 의 정답률 :  0.7521800808693164
LarsCV 의 정답률 :  0.7570138649983484
Lasso 의 정답률 :  0.6855879495660049
LassoCV 의 정답률 :  0.7154057460487299
LassoLars 의 정답률 :  -0.0007982049217318821
LassoLarsCV 의 정답률 :  0.7570138649983484
LassoLarsIC 의 정답률 :  0.754094595988446
LinearRegression 의 정답률 :  0.7521800808693141
LinearSVR 의 정답률 :  0.5624934775374814
MLPRegressor 의 정답률 :  0.628992133536231
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 정답률 :  0.32534704254368274
OrthogonalMatchingPursuit 의 정답률 :  0.5661769106723642
OrthogonalMatchingPursuitCV 의 정답률 :  0.7377665753906506
PLSCanonical 의 정답률 :  -1.7155095545127699
PLSRegression 의 정답률 :  0.7666940310402938
PassiveAggressiveRegressor 의 정답률 :  0.2918736220184288
PoissonRegressor 의 정답률 :  0.8014250117852569
RANSACRegressor 의 정답률 :  0.06983619387475293
RadiusNeighborsRegressor 은 없는 모델
RandomForestRegressor 의 정답률 :  0.8912407528800954
RegressorChain 은 없는 모델
Ridge 의 정답률 :  0.7539303499010775
RidgeCV 의 정답률 :  0.7530092298810112
SGDRegressor 의 정답률 :  -6.780988226337166e+26
SVR 의 정답률 :  0.2868662719877668
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 :  0.796784849729441
TransformedTargetRegressor 의 정답률 :  0.7521800808693141
TweedieRegressor 의 정답률 :  0.6917874063129013
VotingRegressor 은 없는 모델
_SigmoidCalibration 은 없는 모델
'''

# tensorflow _ cnn
# r2 : 0.9304400277059781   ***