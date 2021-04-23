'''
NeuralProphet 설치
> git clone https://github.com/ourownstory/neural_prophet.git
> cd neural_prophet
> pip install .
'''

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

energy_list = ['dangjin_floating','dangjin_warehouse','dangjin','ulsan']
submission = pd.read_csv('../../../data/solar_energy2/sample_submission.csv')

for name in energy_list : 
    print(name)    
    column = name
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

# 계정 xian0105 (학원)
# 파일명 : submission_0410_1.csv
# score : 13.8069400531	

'''
2021-04-10 21:35:59,313; INFO; NP.forecaster; _handle_missing_data; dropped 24 NAN row in 'y'
2021-04-10 21:35:59,335; INFO; NP.config; set_auto_batch_epoch; Auto-set batch_size to 64
2021-04-10 21:35:59,335; INFO; NP.config; set_auto_batch_epoch; Auto-set epochs to 61
2021-04-10 21:36:03,021; INFO; NP.utils_torch; lr_range_test; lr-range-test results: steep: 2.47E-02, min: 3.13E-01
2021-04-10 21:36:03,021; INFO; NP.utils_torch; lr_range_test; learning rate range test selected lr: 1.34E-01
2021-04-10 21:36:45,386; INFO; NP.forecaster; _handle_missing_data; dropped 48 NAN row in 'y'
2021-04-10 21:36:45,398; INFO; NP.config; set_auto_batch_epoch; Auto-set batch_size to 64
2021-04-10 21:36:45,398; INFO; NP.config; set_auto_batch_epoch; Auto-set epochs to 61
2021-04-10 21:36:48,842; INFO; NP.utils_torch; lr_range_test; lr-range-test results: steep: 2.93E-02, min: 4.42E-01
2021-04-10 21:36:48,842; INFO; NP.utils_torch; lr_range_test; learning rate range test selected lr: 1.79E-01
2021-04-10 21:37:30,914; INFO; NP.config; set_auto_batch_epoch; Auto-set batch_size to 64
2021-04-10 21:37:30,915; INFO; NP.config; set_auto_batch_epoch; Auto-set epochs to 61
2021-04-10 21:37:34,340; INFO; NP.utils_torch; lr_range_test; lr-range-test results: steep: 2.77E-02, min: 3.13E-01
2021-04-10 21:37:34,340; INFO; NP.utils_torch; lr_range_test; learning rate range test selected lr: 1.39E-01
2021-04-10 21:38:16,421; INFO; NP.config; set_auto_batch_epoch; Auto-set batch_size to 64
2021-04-10 21:38:16,422; INFO; NP.config; set_auto_batch_epoch; Auto-set epochs to 61
2021-04-10 21:38:19,839; INFO; NP.utils_torch; lr_range_test; lr-range-test results: steep: 2.61E-02, min: 3.13E-01
2021-04-10 21:38:19,839; INFO; NP.utils_torch; lr_range_test; learning rate range test selected lr: 1.37E-01
'''