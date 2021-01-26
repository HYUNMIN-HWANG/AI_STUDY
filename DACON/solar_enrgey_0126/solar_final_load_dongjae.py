# dongjae's
# 그동안 제출한 파일 중에서 점수 가장 잘 나온 5개 파일 묶어서 제출함
# score : 1.84521 (56등)

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

x = []
for i in range(20,25):
    df = pd.read_csv(f'../data/DACON_0126/submission/submission_{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'../data/DACON_0126/submission/submission_{i}.csv', index_col=0, header=0)
for i in range(7776):
    for j in range(9):
        a = []
        for k in range(5):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('../data/DACON_0126/submission/sample_submission25_check.csv') 

