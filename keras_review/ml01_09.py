import numpy as np 
from sklearn.datasets import load_iris 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target

# x > preprocessing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. Modeling
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense 
# from sklearn.svm import LinearSVC, SVC 
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression

# model = LinearSVC()
# model = SVC()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model =  RandomForestClassifier()
# model = LogisticRegression()

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithms :
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, "의 정답률 : ", accuracy_score(y_pred, y_test))

    except :
        print(name, "은 없는 모델")


'''
#3. Train
model.fit(x_train, y_train)

#4. score, Predict
y_pred = model.predict(x_test)
print("y_pred : ", y_pred)

result = model.score(x_test, y_test)
print("model.score : ", result)

acc = accuracy_score(y_pred, y_test)
print("accuracy_score : ", acc)

# model = LinearSVC()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = SVC()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# model.score :  1.0
# accuracy_score :  1.0

# model = KNeighborsClassifier()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# model.score :  1.0
# accuracy_score :  1.0

# model = DecisionTreeClassifier()
# y_pred :  [1 1 1 0 1 1 0 0 0 1 2 2 0 1 2 0 1 1 2 2 0 1 1 2 1 2 0 0 2 2]
# model.score :  0.9333333333333333
# accuracy_score :  0.9333333333333333

# model =  RandomForestClassifier()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

# model = LogisticRegression()
# y_pred :  [1 1 1 0 1 1 0 0 0 2 2 2 0 2 2 0 1 1 2 2 0 1 1 2 1 2 0 0 1 2]
# model.score :  0.9666666666666667
# accuracy_score :  0.9666666666666667

'''

