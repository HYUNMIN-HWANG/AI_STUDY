import numpy as np 
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
import warnings
warnings.filterwarnings('ignore')

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)
kfold = KFold(n_splits=5, shuffle=True, random_state=55)

#2. Modeling
parameters = [
    {"C" : [10, 100, 1000], "kernel" : ["linear", "rbf"]},
    {"C" : [1, 10, 100], "gamma" : [0.001, 0.0001]}
]

model = RandomizedSearchCV(SVC(), parameters, cv=kfold)
score = cross_val_score(model, x, y, cv=kfold)

print("scores : ", score)
#3. Train

#4. Score, predict
