# PCA

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# datasets = load_diabetes()
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

# pca = PCA(n_components=9)
# x2 = pca.fit_transform(x)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum : ", cumsum)

# print("cumsum >= 0.95", cumsum > 0.95)
# d = np.argmax(cumsum>=0.95) + 1
# print("d", d)

pca = PCA(n_components=3)
x2 = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=47)

model = Pipeline([("scaler", MinMaxScaler()), ("model", RandomForestClassifier())])

model.fit(x_train, y_train)

score = model.score(x_test, y_test)
print("model.score : ", score)

y_pred = model.predict(x_test)

acc = accuracy_score(y_pred, y_test)
print("accuracy_score : ", acc)
