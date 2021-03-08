# SelectFromModel
# XGBoostRegressor
# RandomSearchCV

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

#1. DATA
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

#2. Modeling
parameters = [
    {"n_estimators":[90, 100, 110, 120], "learning_rate":[0.1, 0.05, 0.01, 0.005, 0.001], 
    "max_depth":[3,5,7],"colsample_bytree":[0.6, 0.9, 1], "colsample_bylevel":[0.6, 0.7, 0.9]}
]

# model = XGBRegressor(n_jobs=8)
model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters, cv = kf)

#3. Train
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("r2 :", score)


thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)

best_r2 = 0
best_tmp = [0,0] 
best_feature = []

for thresh in thresholds :
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit = True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2 : %2f%%" % (thresh, select_x_train.shape[1],score*100))

    if best_r2 < score :
        best_tmp[0] = thresh
        best_tmp[1] = select_x_train.shape[1]
        best_r2 = score

print("Beat Thresh=%.3f, n=%d, R2 : %2f%%" % (best_tmp[0], best_tmp[1],best_r2*100))

selection = SelectFromModel(model.best_estimator_, threshold=best_tmp[0], prefit=True)

select_x_train = selection.transform(x_train)
print(select_x_train.shape)

select_model = RandomizedSearchCV(XGBRegressor(n_jobs=8), parameters, cv = kf)
select_model.fit(select_x_train, y_train)

select_x_test = selection.transform(x_test)
y_predict = select_model.predict(select_x_test)

score = r2_score(y_test, y_predict)
print("score : ", score)

# Beat Thresh=0.022, n=9, R2 : 91.149753%
# (404, 9)
# score :  0.8973568404348586