# python 딕셔너리에 대해서
# data를 numpy파일로 저장한다.

from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
# print(dataset)
# 딕셔너리로 저장되어 있다. : { 'data' : array(), 'target' : array(), 'frame': None, 'target_names': array() ... }

print(dataset.keys())           
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.frame)            
# None
print(dataset.target_names)     
# ['setosa' 'versicolor' 'virginica'] y로 나올 수 있는 값
print(dataset['DESCR'])         
# 데이터에 관한 설명
print(dataset['feature_names']) 
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] < 'feature' 이름
print(dataset.filename)         
# C:\Users\ai\Anaconda3\lib\site-packages\sklearn\datasets\data\iris.csv < csv 저장되어 있는 위치 

# 딕셔너리를 사용해 데이터 불러오기
# x = dataset.data
# y = dataset.target
x_data = dataset['data']   # FLOAT 형 리스트로 저장되어 있음
y_data = dataset['target'] # INT 형 리스트로 저장되어 있음
# print(x)
# print(y)

print(type(x_data), type(y_data))       # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <-- numpy 형식
np.save('../data/npy/iris_x.npy',arr=x_data) # numpy 형식으로 저장하겠다. 저장한건 x_data
np.save('../data/npy/iris_y.npy',arr=y_data) # numpy 형식으로 저장하겠다. 저장한건 y_data

