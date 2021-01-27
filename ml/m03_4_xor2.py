# XOR 해결
# LinearSVC에서 개선된 것 >> SVC >> acc :  1.0
# accuracy_score    # 정확도

from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score

#1. DATA (XOR)
x_data = [[0,0], [1,0], [0,1], [1,1]]
y_data = [0, 1, 1, 0]

#2. Modeling
# model = LinearSVC()
model = SVC()

#3. Train
model.fit(x_data, y_data)

#4. evaluate, Predict

y_pred = model.predict(x_data)      
print(x_data, "의 예측결과 : ", y_pred)  # [[0, 0], [1, 0], [0, 1], [1, 1]] 의 예측결과 :  [0 1 1 0]

result = model.score(x_data, y_data)
print("model.score : ", result)         # model.score :  1.0

acc = accuracy_score(y_data, y_pred)
print("accuracy_score : ", acc)         # accuracy_score : 1.0