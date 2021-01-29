from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target

x_pd = pd.DataFrame(x, columns=dataset['feature_names'])
x = x_pd.drop(['sepal width (cm)'], axis=1)
x = x.to_numpy()

x_train, x_test, y_train, y_test = \
    train_test_split(x, y, train_size=0.8, random_state=47)

#2. Modeling
# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()
model = GradientBoostingClassifier()

#3. Train
model.fit(x_train, y_train)

#4. Screo, Predict
score = model.score(x_test, y_test)
print(model.feature_importances_)
print("acc :", score )

'''import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset (model) :
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,)
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()'''