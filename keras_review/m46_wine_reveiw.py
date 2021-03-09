import pandas as pd
import numpy as np

#1. DATA
wine = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';', index_col=None)
print(wine.shape)   # (4898, 12)

print(np.unique(wine['quality']))   # [3 4 5 6 7 8 9]

count_data = wine.groupby('quality')['quality'].count()
print(count_data)
'''
quality
3      20
4     163
5    1457
6    2198
7     880
8     175
9       5
Name: quality, dtype: int64
'''


wine = wine.values

x = wine[:,:-1]
y = wine[:,-1]
print(x.shape, y.shape) # (4898, 11) (4898,)

newlist = []
for i in list(y) :
    if i <= 4 :
        newlist += [0]
    elif i <= 7 :
        newlist += [1]
    else :
        newlist += [2]

y = newlist
print(np.unique(y)) # [0 1 2]


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

scale = RobustScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)

print(x_train.shape, x_test.shape)  # (3918, 11) (980, 11)

#2. Modeling
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

#3. Train
model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score : ", score)
# score :  0.9540816326530612


