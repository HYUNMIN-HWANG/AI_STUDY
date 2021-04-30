# https://www.kaggle.com/remekkinas/ensemble-learning-meta-classifier-for-stacking/output 

import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.simplefilter('ignore')

from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from mlxtend.classifier import StackingCVClassifier
import shap

RANDOM_SEED = 2021
PROBAS = True
FOLDS = 8
N_ESTIMATORS = 1000

TARGET = 'Survived'

def outliers (data) :
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)

    return np.where((data>upper_bound) | (data < lower_bound))

# 데이터 로드하기
train_df = pd.read_csv('E:\data\kaggle_tabular/train.csv')
test_df = pd.read_csv('E:\data\kaggle_tabular/test.csv')
submission = pd.read_csv('E:\data\kaggle_tabular/sample_submission.csv')

# Pseudo labels taken from great BIZEN notebook: https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble
pseudo_labels = pd.read_csv('E:\data\kaggle_tabular/pseudo_label.csv')
test_df[TARGET] = pseudo_labels[TARGET]

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# print(all_df.shape) # (200000, 12)
# print(all_df.head()) 


submission['submit_1'] = pd.read_csv("E:\\data\\kaggle_tabular\\submission\\submission_0430_5.csv")[TARGET]
submission['submit_2'] = pd.read_csv("E:\\data\\kaggle_tabular\\submission\\submission_0429_voting6.csv")[TARGET]
submission['submit_2'] = pd.read_csv("E:\\data\\kaggle_tabular\\submission\\submission_0430_3.csv")[TARGET]

submission[TARGET] = (submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis=1) >= 2).astype(int)
submission[['PassengerId', TARGET]].to_csv("E:\\data\\kaggle_tabular\\submission_0430_f4.csv", index = False)

# submission_0430_f4.csv 
# score 0.81730

import winsound as sd
def beepsound():
    fr = 800    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()
