# XGBoost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpt as np


#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
kf = KFold(n_splits=5, shuffle=True, random_state=47)

#2. Modeling
parameters = [
    {"n-estimators" : [10, 20, 30], "learning_rate":[0.1, 0.2, 0.3]},
    {"n-estimators" : [10, 20, 30], "max_depth" : [4, 5, 6]},
    {"n-estimators" : [10, 20, 30], "colsample_bytree":[0.3, 0.6, 0.9]}
]

# model = XGBClassifier(n_jobs = 8, use_label_emcoder=False)
model = GridSearchCV(XGBClassifier(n_jobs=-1, use_label_encoder=False), parameters, cv=kf)
model = RandomizedSearchCV(XGBClassifier(n_jobs=-1, use_label_encoder=False), parameters, cv=kf)

#3. Train
model.fit(x_train, y_train, eval_metric='logloss')

#4. Score, Predict
print("feauture_importances ", model.feature_importances_)

score = model.score(x_test, y_test)
print("model.score : ", score)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("accuracy_score : ", acc)

# import matplotlib.pyplot as plt
# plot_importance(model)
# plt.show()
