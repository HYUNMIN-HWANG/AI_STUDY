# 모든 분류모델 kfold 적용

# 모든 회귀모델

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.datasets import load_diabetes
import warnings

warnings.filterwarnings('ignore')

dataset = load_diabetes()
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
 [0.26990101 0.40174264 0.53667169 0.59380198 0.5214386 ]
AdaBoostRegressor 의 정답률 : 
 [0.35177473 0.29744221 0.36792513 0.45881116 0.48225577]
BaggingRegressor 의 정답률 : 
 [0.50539866 0.2595259  0.31842837 0.38019751 0.41184283]
BayesianRidge 의 정답률 : 
 [0.55727615 0.34913012 0.50817389 0.53592623 0.33823191]
CCA 의 정답률 :
 [0.52344818 0.48850604 0.34551805 0.26079479 0.24964336]
DecisionTreeRegressor 의 정답률 : 
 [-0.23462427  0.13904178  0.05554157 -0.26020959 -0.53070111]
DummyRegressor 의 정답률 :
 [-1.61727145e-02 -1.13125108e-05 -6.36088398e-02 -1.81081212e-02
 -4.77960860e-06]
ElasticNet 의 정답률 :
 [-0.00671502  0.00288282  0.00843176 -0.03159124 -0.00591849]
ElasticNetCV 의 정답률 : 
 [0.37064374 0.50244239 0.31978893 0.40482563 0.4858813 ]
ExtraTreeRegressor 의 정답률 :
 [ 0.05677675  0.13287308 -0.28880453 -0.1144689  -0.18811498]
ExtraTreesRegressor 의 정답률 : 
 [0.51521573 0.35874372 0.51358166 0.45774394 0.24199934]
GammaRegressor 의 정답률 :
 [ 0.00564285  0.00642025 -0.00221639  0.00266084  0.00453754]
GaussianProcessRegressor 의 정답률 : 
 [-20.09723092 -17.29732511  -9.97456304 -26.38090763  -8.50131887]
GeneralizedLinearRegressor 의 정답률 :
 [ 0.0066249   0.00425118 -0.02568246 -0.09016868  0.0057    ]
GradientBoostingRegressor 의 정답률 : 
 [0.19340778 0.40824903 0.34890439 0.52322464 0.27056373]
HistGradientBoostingRegressor 의 정답률 : 
 [0.35718631 0.27651413 0.33739889 0.19096991 0.49676607]
HuberRegressor 의 정답률 : 
 [0.57561588 0.36918624 0.43870172 0.24407085 0.5068381 ]
IsotonicRegression 의 정답률 :
 [nan nan nan nan nan]
KNeighborsRegressor 의 정답률 : 
 [0.07468706 0.27453102 0.43912049 0.3703535  0.36803105]
KernelRidge 의 정답률 :
 [-3.2093988  -3.68045048 -3.51714406 -3.88799109 -3.56179792]
Lars 의 정답률 : 
 [ 0.46523074  0.44304969  0.41880123 -5.96038966 -0.80914665]
LarsCV 의 정답률 : 
 [0.49052382 0.44359017 0.45609584 0.35216679 0.51676512]
Lasso 의 정답률 :
 [0.37005224 0.31186336 0.37043935 0.28311248 0.30907015]
LassoCV 의 정답률 : 
 [0.46086656 0.54352636 0.49081574 0.224811   0.49887953]
LassoLars 의 정답률 :
 [0.37225771 0.40637539 0.31711539 0.34701159 0.37685575]
LassoLarsCV 의 정답률 : 
 [0.4428109  0.53785625 0.52901437 0.47148095 0.38997143]
LassoLarsIC 의 정답률 : 
 [0.53467017 0.41571727 0.30330921 0.42187801 0.51375093]
LinearRegression 의 정답률 :
 [0.49784547 0.20727154 0.46426349 0.49947642 0.57817841]
LinearSVR 의 정답률 :
 [-0.29841749 -0.47821619 -0.81637862 -0.19282293 -0.58973633]
MLPRegressor 의 정답률 : 
 [-2.59297281 -2.4419873  -2.94250391 -3.23106237 -3.27439991]
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
 [0.1244304  0.07386601 0.16488851 0.11923722 0.12930926]
OrthogonalMatchingPursuit 의 정답률 : 
 [0.3550378  0.36380149 0.32945034 0.33887182 0.24280246]
OrthogonalMatchingPursuitCV 의 정답률 : 
 [0.42454428 0.32072998 0.56333609 0.49941531 0.49776534]
PLSCanonical 의 정답률 :
 [-1.89062941 -0.97001892 -1.15365805 -1.43323528 -1.06857202]
PLSRegression 의 정답률 : 
 [0.55542736 0.196808   0.4272869  0.54590408 0.54946559]
PassiveAggressiveRegressor 의 정답률 :
 [0.46597279 0.38536563 0.43929339 0.42243114 0.47455668]
PoissonRegressor 의 정답률 : 
 [0.35983282 0.24375785 0.29926829 0.33512486 0.36034668]
RANSACRegressor 의 정답률 : 
 [ 0.20503017 -1.68900085 -0.29241422  0.32961564  0.17325176]
RadiusNeighborsRegressor 의 정답률 :
 [-0.03778226 -0.00466387 -0.00047966 -0.01111743 -0.0004375 ]
RandomForestRegressor 의 정답률 : 
 [0.49048894 0.33796899 0.50019479 0.38036406 0.54466647]
RegressorChain 은 없는 모델
Ridge 의 정답률 :
 [0.3193029  0.47288315 0.38381491 0.40961568 0.34194789]
RidgeCV 의 정답률 :
 [0.33791191 0.50303187 0.48819847 0.55253656 0.43286942]
SGDRegressor 의 정답률 : 
 [0.40780932 0.47321002 0.35643436 0.36779125 0.33157817]
SVR 의 정답률 : 
 [ 0.14554429  0.19127361 -0.09642766  0.1337614   0.09143987]
StackingRegressor 은 없는 모델
TheilSenRegressor 의 정답률 : 
 [0.44380371 0.28576698 0.58883635 0.50405728 0.37349988]
TransformedTargetRegressor 의 정답률 :
 [0.55816367 0.45418999 0.36598453 0.44855772 0.49817129]
TweedieRegressor 의 정답률 :
 [0.00712236 0.00558244 0.00664906 0.00592361 0.00425261]
VotingRegressor 은 없는 모델
_SigmoidCalibration 의 정답률 :
 [nan nan nan nan nan]
'''

# tensorflow _ dense (modelcheckpoint)
# r2 : 0.5510438539208138   ***