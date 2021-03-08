import numpy as np

from sklearn.datasets import load_boston 

dataset = load_boston()

#1. DATA

x = dataset.data
y = dataset.target      # target : x와 y 가 분리한다.
print(x.shape, y.shape) # (506, 13) (506,)

# preprocess

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()         # min 0.0 max 1.0
# scaler = StandardScaler()       # min -3.9071933049810337 max 9.933930601860268
# scaler = RobustScaler()         # min -18.76100251828754 max 24.678376790228196
# scaler = QuantileTransformer()  # min 0.0 max 1.0
# scaler = QuantileTransformer(output_distribution="normal")  # min -5.199337582605575 max 5.19933758270342
# scaler = MaxAbsScaler()         # min 0.0 max 1.0
scaler = PowerTransformer(method='yeo-johnson') # min -4.478632778203448 max 3.6683978597124267
# scaler = PowerTransformer(method='box-cox') # 음수 데이터도 있기 때문에 안 됨
scaler.fit(x)       
x = scaler.transform(x)

print("min", np.min(x),"max", np.max(x))

