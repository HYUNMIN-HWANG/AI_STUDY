# Pipe lind
import numpy as np
from sklearn.datasets import load_boston

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')
import pandas as pd

#1. DATA
dataset = load_boston()
# print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
x_feature_names = ['LSTAT', 'DIS', 'RM']
df_trim = df[x_feature_names]

x = df_trim.to_numpy()
# x = dataset.data
y = dataset.target



x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8,shuffle=True, random_state=44)
kf = KFold(n_splits=5, shuffle=True, random_state=47)

#2. Modeling
# pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestRegressor())])
# parameters = [
#     {"model__n_estimators" : [10, 100], "model__max_depth" : [6, 8]},
#     {"model__n_estimators" : [100, 200], "model__min_samples_leaf" : [3, 6]}
# ]

# pipe = make_pipeline(MinMaxScaler(), RandomForestRegressor())
# parameters = [
#     {'randomforestregressor__n_estimators' : [10, 100], 'randomforestregressor__max_depth' : [6, 8]},
#     {'randomforestregressor__n_estimators' : [100, 200], 'randomforestregressor__min_samples_leaf' : [3, 6]}
# ]

# model = GridSearchCV(pipe, parameters, cv=kf)
# model = RandomizedSearchCV(pipe, parameters, cv=kf)

# model = DecisionTreeRegressor(max_depth=4)
# model = RandomForestRegressor(max_depth=4)
model = GradientBoostingRegressor()

#3. Train
model.fit(x_train, y_train)

#4. Score, Predict
score = model.score(x_test, y_test)
print("model.score : ", score)  # model.score :  0.8481855458478802

print("feature_importances : ", model.feature_importances_)

