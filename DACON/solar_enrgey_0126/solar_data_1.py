import numpy as np
import pandas as pd


def drop_3col(data) :
    data = data.drop(['Day', 'Hour','Minute'], axis=1)
    return data

def split_xy(dataset, time_steps, y_column) :
    x, y = list(), list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column
        if y_end_number > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, :]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

def split_xy2(dataset, time_steps) :
    x = list()
    for i in range(len(dataset)) :
        x_end_number = i + time_steps
        if time_steps > len(dataset) :
            break
        tmp_x = dataset[i:x_end_number, : ]
        x.append(tmp_x)
    return np.array(x)




# train 데이터 불러오기 >> x_train
train_df = pd.read_csv('../data/DACON_0126/train/train.csv')
# print(train.columns)    # Index(['Day', 'Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
# print(train.shape)      # (52560, 9)
train_df = drop_3col(train_df)
# print(x_train.shape)      # (52560, 6)
dataset = train_df.to_numpy()

x, y = split_xy(dataset, 336, 96)
print(x.shape)  # (52129, 336, 6)
print(y.shape)  # (52129, 96, 6)

# test 데이터 불러오기 >> x_pred
df_pred = []
for i in range(81):
    file_path = '../data/DACON_0126/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = drop_3col(temp)
    df_pred.append(temp)

df_pred = pd.concat(df_pred)
# print(x_pred.shape) # (27216, 6)
pred_dataset = df_pred.to_numpy()

x_pred = split_xy2(pred_dataset, 336)
print(x_pred.shape)  # (27216,)
# print(x_pred)

# submission 데이터 불러오기 ; y_pred 값 합쳐서 넣어야 함
sub = pd.read_csv('../data/DACON_0126/sample_submission.csv')

