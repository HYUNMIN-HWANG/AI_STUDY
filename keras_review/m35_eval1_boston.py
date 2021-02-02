# eval_set

from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(\
    x, y, train_size=0.8, shuffle=True, random_state=47)

#2. Modeling
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, n_jobs=8)

#3. Train
model.fit(x_train, y_train, verbose=1, eval_metric=['rmse','logloss'],\
     eval_set=[(x_train, y_train),(x_test, y_test)], early_stopping_rounds=10)

#4. Score, Predict
score = model.score(x_test, y_test)
print("score : ", score)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2 : ", r2)

print("================")
result = model.evals_result()
# print(result)

import pickle, joblib
# pickle.dump(model, open('../data/xgb_save/m39_pickle_data', 'wb'))
# joblib.dump(model, '../data/xgb_save/m39_joblib_data')
# model.save_model('../data/xgb_save/m39_xgb_data')


# model2 = pickle.load(open('../data/xgb_save/m39_pickle_data','rb'))
# model2 = joblib.load('../data/xgb_save/m39_joblib_data','rb')
model2 = XGBRegressor()
model2.load_model('../data/xgb_save/m39_xgb_data')
r22 = model2.score(x_test, y_test)
print("r22 :", r22)
