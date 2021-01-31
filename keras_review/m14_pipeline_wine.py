# wine

import numpy as np
from sklearn.datasets import load_wine

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

#1. DATA
dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)
kf = KFold(n_splits=4, shuffle=True, random_state=47)

'''
allAlgorithm = all_estimators(type_filter = 'classifier')
for (name, algorithm) in allAlgorithm :
    try : 
        model = algorithm()
        # model.fit(x_train, y_train)
        score = cross_val_score(model, x_train, y_train, cv=kf)
        print(name, ' acc kf : ', score)
        y_pred = model.predict(x_test)

    except :
        print(name, "은 없는 모델")
'''

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# pipe = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestClassifier())])
# parameters = [
#     {'model__n_estimators':[100, 200], 'model__max_depth':[6, 8, 10]},
#     {'model__max_depth' : [2, 4, 6], 'model__min_samples_leaf':[3,7,9]}
# ]

pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
parameters = [
    {'randomforestclassifier__n_estimators' : [100, 200], 'randomforestclassifier__max_depth':[6, 8]},
    {'randomforestclassifier__max_depth':[2,4], 'randomforestclassifier__min_samples_leaf':[3, 5]}
]

#2. Modeling
# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kf)
# model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kf)
model = RandomizedSearchCV(pipe, parameters, cv=kf)


#3. Train
model.fit(x_train, y_train)

#4. Score, Predict

# print("최적의 매개변수 : ", model.best_estimator_)
# print("feature_importances :", model.feature_importances_ )

y_pred = model.predict(x_test)
print("y_pred : ", y_pred)

# score = model.score(x_test, y_test)
# print("score : ", score)

score = cross_val_score(model, x_train, y_train, cv=kf)
print("score : ", score)

acc = accuracy_score(y_test, y_pred)
print("acc : ", acc)

