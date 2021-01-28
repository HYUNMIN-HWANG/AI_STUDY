# 모든 분류모델 kfold 적용

# 모든 회귀모델

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')    # type_filter='regressor' : 회귀형 모델 전체를 불러온다.

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
ARDRegression 의 정답률 : 
 [0.60160215 0.67555301 0.67574986 0.72412908 0.79494722]
AdaBoostRegressor 의 정답률 : 
 [0.65665232 0.87392171 0.84075669 0.90341913 0.85865114]
BaggingRegressor 의 정답률 : 
 [0.73760773 0.78973001 0.91143702 0.83836575 0.79200249]
BayesianRidge 의 정답률 : 
 [0.73347776 0.53653676 0.60397929 0.75414185 0.70324877]
CCA 의 정답률 :
 [0.71743851 0.70324054 0.77526325 0.27279564 0.69342033]
DecisionTreeRegressor 의 정답률 : 
 [0.59453539 0.76623963 0.57673548 0.75472021 0.77861922]
DummyRegressor 의 정답률 : 
 [-6.68912002e-03 -6.55015875e-03 -6.38288751e-07 -7.03841599e-03
 -9.12365947e-03]
ElasticNet 의 정답률 :
 [0.69053111 0.55232187 0.71839335 0.66954941 0.65038066]
ElasticNetCV 의 정답률 : 
 [0.64891955 0.74633307 0.621986   0.51454614 0.59430482]
ExtraTreeRegressor 의 정답률 : 
 [0.35438475 0.6926151  0.74937803 0.49430472 0.84237716]
ExtraTreesRegressor 의 정답률 : 
 [0.85375471 0.92391475 0.90494236 0.84092221 0.83853133]
GammaRegressor 의 정답률 :
 [-1.78233123e-06 -7.64919332e-03 -1.43240724e-03 -2.11221523e-02
 -2.48307545e-04]
GaussianProcessRegressor 의 정답률 : 
 [-5.96116379 -5.83329332 -6.26537816 -6.22927602 -6.06756322]
GeneralizedLinearRegressor 의 정답률 : 
 [0.71696848 0.65655062 0.57657243 0.60453222 0.65715714]
GradientBoostingRegressor 의 정답률 : 
 [0.91045409 0.82622283 0.90072821 0.85952127 0.73303026]
HistGradientBoostingRegressor 의 정답률 : 
 [0.83029963 0.83972823 0.85605776 0.72530907 0.90014365]
HuberRegressor 의 정답률 : 
 [0.63598941 0.69199276 0.71127094 0.57874212 0.48813735]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 : 
 [0.47392608 0.51043588 0.55640988 0.44626624 0.36164167]
KernelRidge 의 정답률 :
 [0.50361123 0.5246239  0.70894791 0.73868095 0.67068349]
Lars 의 정답률 : 
 [0.61526584 0.53257044 0.78189962 0.64757745 0.736598  ]
LarsCV 의 정답률 : 
 [0.68800425 0.64362704 0.67256063 0.74064466 0.71479809]
Lasso 의 정답률 :
 [0.70618056 0.63623189 0.64508277 0.59210073 0.65773632]
LassoCV 의 정답률 : 
 [0.69990589 0.69153455 0.69582681 0.59055225 0.6551674 ]
LassoLars 의 정답률 :
 [-0.0098069  -0.05084128 -0.00472319 -0.02417096 -0.0277105 ]
LassoLarsCV 의 정답률 : 
 [0.69583047 0.7273157  0.56663214 0.70535301 0.77615901]
LassoLarsIC 의 정답률 : 
 [0.79469191 0.65146243 0.58301929 0.68800628 0.76436384]
LinearRegression 의 정답률 :
 [0.78021488 0.6338909  0.71129664 0.70250723 0.58216212]
LinearSVR 의 정답률 : 
 [0.50934333 0.47108923 0.46697559 0.55306335 0.64809279]
MLPRegressor 의 정답률 : 
 [ 0.55528036  0.64879873 -1.66861391  0.66679043  0.29780409]
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 의 정답률 :
 [nan nan nan nan nan]
MultiTaskElasticNetCV 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLasso 의 정답률 :
 [nan nan nan nan nan]
MultiTaskLassoCV 의 정답률 : 
 [nan nan nan nan nan]
NuSVR 의 정답률 : 
 [0.10857987 0.2405836  0.16024698 0.30283446 0.24246021]
OrthogonalMatchingPursuit 의 정답률 :
 [0.49171879 0.57684524 0.61274711 0.44804976 0.48271186]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.69413916 0.51194338 0.6961425  0.70649016 0.69009397]
PLSCanonical 의 정답률 :
 [-2.670654   -1.60499237 -1.92638695 -3.23898978 -2.02259824]
PLSRegression 의 정답률 :
 [0.64917375 0.54495119 0.74528213 0.6862031  0.72717102]
PassiveAggressiveRegressor 의 정답률 : 
 [-0.63680378  0.20221639 -0.08221594  0.10847308  0.06040225]
PoissonRegressor 의 정답률 : 
 [0.72241731 0.77109383 0.75825308 0.7639958  0.70821287]
RANSACRegressor 의 정답률 : 
 [-0.77883184  0.29187712  0.49203799  0.43457158  0.48794676]
RadiusNeighborsRegressor 은 없는 모델
RandomForestRegressor 의 정답률 : 
 [0.88618245 0.70937334 0.88394233 0.84359527 0.87674749]
RegressorChain 은 없는 모델
Ridge 의 정답률 :
 [0.73326021 0.65417559 0.75000998 0.62516181 0.69075466]
RidgeCV 의 정답률 : 
 [0.75016869 0.7516625  0.58541646 0.73919504 0.6957544 ]
 [-4.64339247e+25 -5.29704790e+26 -3.09084387e+25 -3.68321560e+26
 -4.06560178e+26]
SVR 의 정답률 :
 [0.19082573 0.11793542 0.21215974 0.06315695 0.24078789]
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 :
 [0.63755265 0.77030426 0.67183695 0.69563236 0.57996995]
TransformedTargetRegressor 의 정답률 :
 [0.72629967 0.48162143 0.78555647 0.66779153 0.78539268]
TweedieRegressor 의 정답률 :
 [0.62821119 0.62019577 0.61876969 0.72264278 0.61189231]
VotingRegressor 은 없는 모델
_SigmoidCalibration 의 정답률 :
 [nan nan nan nan nan]
'''

# tensorflow _ cnn
# r2 : 0.9304400277059781   ***