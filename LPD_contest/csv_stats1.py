import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from scipy import stats


x = []
for i in range(1,12):
    if i != 11 : 
        df = pd.read_csv(f'../data/LPD_competition/submission/mean_csv/check/sub1 ({i}).csv', index_col=0, header=0)
        data = df.to_numpy()
        x.append(data)
x = np.array(x)

# print(x.shape)
a= []
# df = pd.read_csv(f'../data/lotte/csv/answer ({i}).csv', index_col=0, header=0)
for i in range(72000):
    for j in range(1):
        b = []
        for k in range(8):
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0]) 
# a = np.array(a)
# a = a.reshape(72000,4)

# print(a)

submission = pd.read_csv('../data/LPD_competition/sample.csv', index_col=0)
submission['prediction'] = np.array(a)
submission.to_csv('../data/LPD_competition/submission/mean_csv/check/check_0325_11.csv', index=True)

# check 전체 : 81.526
# 1 제거 : 81.157
# 2 제거 : 81.533
# 3 제거 : 81.926
# 4 제거 : 81.912
# 5 제거 : 81.515
# 6 제거 : 82.121   ***
# 7 제거 : 82.117
# 8 제거 : 81.492
# 9 제거 : 81.526
# 10 제거 : 81.526
# 11 제거 : 81.526
