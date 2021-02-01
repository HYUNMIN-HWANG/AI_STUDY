import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000, 28*28)
print(x.shape)
x = x/255.

y = np.append(y_train, y_test, axis=0)
print(y.shape)

# pca = PCA()
# pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)

# d = np.argmax(cumsum >= 0.95)+1
# print("cumsum >= 0.95", cumsum > 0.95)
# print("d : ", d)    # 154

pca = PCA(n_components=154)
x2 = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x2, y, train_size=0.8, shuffle=True, random_state=47)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

#2. Modeling
model = XGBClassifier(n_jobs=8, use_label_encoder=False)

#3. Train
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# es = EarlyStopping(monitor='acc', patience=5, mode='max')
# lr = ReduceLROnPlateau(monitor='acc', factor=0.4, patience=10, verbose=1, mode='max')

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, eval_metric='logloss')

#4. Score, Predict
acc = model.score(x_test, y_test)
print("acc ", acc)

y_pred = model.predict(x_test)
print("y_pred ", np.argmax(y_pred, axis=1))

