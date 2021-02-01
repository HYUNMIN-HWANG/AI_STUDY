# PCA : 차원축소, 컬럼 재구성

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (442, 10) (442,)

pca = PCA(n_components=9)  # n_components= : 내가 최종적으로 자르고 싶은 컬럼 수를 넣는다.
x2 = pca.fit_transform(x)  # 전처리 fit과 transform 한꺼번에 한다.

print(x2)
print(x2.shape)            # (442, 7) >> 컬럼을 압축시켰다. 컬럼 재구성됨

pca_EVR = pca.explained_variance_ratio_ # 컬럼이 어느 정도의 변화율을 보여주었는지 보여준다.
print(pca_EVR)
print(sum(pca_EVR)) 
# n_components=7개 압축률 : 0.9479436357350414
# n_components=8개 압축률 : 0.9913119559917797
# n_components=9개 압축률 : 0.9991439470098977  * 9개 컬럼을 사용했을 때 가장 성능이 좋다.
