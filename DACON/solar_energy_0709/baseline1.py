import pandas as pd
import numpy as np
from datetime import datetime
from neuralprophet import NeuralProphet
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def convert_time(x):
    Ymd, HMS = x.split(' ')
    H, M, S = HMS.split(':')
    H = str(int(H)-1)
    HMS = ':'.join([H, M, S])
    return ' '.join([Ymd, HMS])


# 데이터 불러오기
train_data = pd.read_csv('../../../data/solar_energy2/energy.csv')
# 시간 변환
train_data['time'] = train_data['time'].apply(lambda x:convert_time(x))

column = 'dangjin_floating'
df = pd.DataFrame()
df['ds'] = train_data['time']
df['y'] = train_data[column]

# 모델 설정
model = NeuralProphet()
# 훈련
loss = model.fit(df, freq="H")
# 예측용 데이터 프레임 만들기
df_pred = model.make_future_dataframe(df, periods=18000)
# 예측
predict = model.predict(df_pred)

submission = pd.read_csv('../../../data/solar_energy2/sample_submission.csv')

# 2021-02-01 ~ 2021-03-01
predict_1 = predict.copy()
predict_1 = predict_1.query('ds >= "2021-02-01 00:00:00"')
predict_1 = predict_1.query('ds < "2021-03-01 00:00:00"')

# 2021-06-09 ~ 2021-07-09
predict_2 = predict.copy()
predict_2 = predict_2.query('ds >= "2021-06-09 00:00:00"')
predict_2 = predict_2.query('ds < "2021-07-09 00:00:00"')

# 제출 파일 업데이트
submission[column] = list(predict_1['yhat1']) + list(predict_2['yhat1'])

print(submission.head())

submission.to_csv('../../../data/solar_energy2/submission_0410_1.csv', index=False)
submission.to_csv('../../../data/solar_energy2/submission_0410_2.csv', index=True)
