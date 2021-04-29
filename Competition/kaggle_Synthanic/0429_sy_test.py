# https://www.kaggle.com/remekkinas/ensemble-learning-meta-classifier-for-stacking/output 
import pandas as pd
import numpy as np
import random
import os

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import lightgbm as lgb
import catboost as ctb
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier

import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter('ignore')
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

TARGET = 'Survived'

N_ESTIMATORS = 200
N_SPLITS = 6
SEED = 2021
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 100

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
set_seed(SEED)

# 데이터 불러오기
train_df = pd.read_csv('E:\data\kaggle_tabular/train.csv')
test_df = pd.read_csv('E:\data\kaggle_tabular/test.csv')
submission = pd.read_csv('E:\data\kaggle_tabular/sample_submission.csv')
test_df[TARGET] = pd.read_csv('E:\data\kaggle_tabular/pseudo_label.csv')[TARGET]

# print(train_df.shape, test_df.shape, submission.shape)  # (100000, 12) (100000, 12) (100000, 2)

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)


test1 = pd.read_csv('E:\data\kaggle_tabular/submission_0429_test1.csv')
# print(test1.head())
# print(test1[[col for col in test1.columns if col.startswith('submit_')]].sum(axis = 1).value_counts())
# 가로끼리 더한다.
test1[TARGET] = (test1[[col for col in test1.columns if col.startswith('submit_')]].sum(axis=1) >= 2).astype(int)
# 가로끼리 더한 것 중 2이상인 것들만 남는다. == 2개 이상의 모델에서 1이라고 예측한 것들만 남는다.
# print(test1)
# test1[[col for col in test1.columns if col.startswith('submit_')]].sum(axis = 1).value_counts()
print(test1[TARGET].mean())
test2 = pd.read_csv('E:\data\kaggle_tabular/submission_0429_test2.csv')
test3 = pd.read_csv('E:\data\kaggle_tabular/submission_0429_test3.csv')

