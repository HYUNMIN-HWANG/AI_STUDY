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

print(all_df.shape) # (200000, 12)
print(all_df.head()) 


# 데이터 전처리
# Age fillna with mean age for each class
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

# Cabin, fillna with 'X' and take first letter
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())
# print(all_df['Cabin'].describe())
# print(all_df.groupby(['Survived', 'Cabin'])[['PassengerId']].count())

# Ticket, fillna with 'X', split string and take first split 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
all_df['Fare'] = np.log1p(all_df['Fare'])
print(all_df['Fare'].describe())

# Embarked, fillna with 'X' value
all_df['Embarked'] = all_df['Embarked'].fillna('X')

# Name, take only surnames
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])

# col : 'SibSp' + 'Parch' = 'Family'    # 가족과 관련된 부분 다 합쳐보자 !!!!!
all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] + 1
all_df['IsAlone'] = np.where(all_df['FamilySize'] <= 1, 1, 0)
all_df = all_df.drop(['SibSp','Parch'], axis=1)

# fare 이상치 삭제 후, 위에서 nan 값 채웠던 방식과 똑같이 이상치 채우기
fare_outlier = outliers(all_df['Fare'])[0]  
print(all_df.loc[fare_outlier, 'Fare'])
print(all_df['Fare'].describe())
all_df.loc[fare_outlier, 'Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
all_df.loc[fare_outlier, 'Fare'] = np.log1p(all_df.loc[fare_outlier, 'Fare'])

# not_outlier = []
# for i in all_df.index :
#     if i not in fare_outlier :
#         not_outlier.append(i)
# print(all_df.loc[not_outlier, 'Fare'])

# plt.figure(figsize=(12,8))
# sns.boxplot(data=all_df['Fare'], color='yellow')
# plt.show()

# print(all_df.head(5))
# print(all_df.groupby(['Survived', 'FamilySize'])[['PassengerId']].count())
# print(all_df.groupby(['Survived', 'IsAlone'])[['PassengerId']].count())
# print(all_df.groupby(['Survived', 'Pclass'])[['PassengerId']].count())
# print(all_df.groupby(['Survived', 'Fare'])[['PassengerId']].count())


# label_cols = ['Name', 'Ticket', 'Sex']    # Name이 왜 필요하지 ?? 빼보자 !!!!!
# onehot_cols = ['Cabin', 'Embarked']
# numerical_cols = ['Pclass', 'Age', 'FamilySize', 'IsAlone', 'Fare']

label_cols = ['Pclass', 'Ticket', 'Sex']        # 'Pclass'를 label columns로 이동 
onehot_cols = ['Cabin', 'Embarked']
numerical_cols = ['Age', 'FamilySize', 'IsAlone', 'Fare']

def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)

scaler = StandardScaler()

onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = pd.DataFrame(scaler.fit_transform(all_df[numerical_cols]), columns=numerical_cols)
target_df = all_df[TARGET]

all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df, target_df], axis=1)
# all_df = all_df.drop(['Cabin_T'], axis=1)   # 중요도가 덜 한 컬럼을 삭제한다.

print(all_df.head(5))
print(all_df.shape) # (200000, 20)


# Modeling
## 모델 정의하기
lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': N_ESTIMATORS,
    'objective': 'binary',
    'random_state': RANDOM_SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}

ctb_params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': RANDOM_SEED,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': N_ESTIMATORS,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED
}

cl1 = KNeighborsClassifier(n_neighbors = 1)
cl2 = RandomForestClassifier(**rf_params)
cl3 = GaussianNB()
cl4 = DecisionTreeClassifier(max_depth = 5)
cl5 = CatBoostClassifier(**ctb_params, verbose = None, logging_level = 'Silent')
cl6 = LGBMClassifier(**lgb_params)
cl7 = ExtraTreesClassifier(bootstrap=False, criterion='entropy', max_features=0.55, min_samples_leaf=8, min_samples_split=4, n_estimators=100) # Optimized using TPOT
cl8 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = RANDOM_SEED)

mlr = LogisticRegression()
scl = StackingCVClassifier(classifiers= [cl2, cl5, cl6, cl7], #[cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8]
                            meta_classifier = mlr, # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)

NUM_CLAS = 5
classifiers = {"RandomForest": cl2,
               "CatBoost": cl5,
               "LGBM": cl6,
               "ExtraTrees": cl7,
               "Stacked": scl}

# DATASET
X = all_df.drop([TARGET], axis = 1)
y = all_df[TARGET]

print (f'X:{X.shape} y: {y.shape} \n')  # X:(200000, 19) y: (200000,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = RANDOM_SEED)
print (f'X_train:{X_train.shape} y_train: {y_train.shape}') # X_train:(160000, 19) y_train: (160000,)
print (f'X_test:{X_test.shape} y_test: {y_test.shape}') # X_test:(40000, 19) y_test: (40000,)

test = all_df[len(train_df):].drop([TARGET], axis = 1)
print (f'test:{test.shape}')    # test:(100000, 19)

# Train Classifier
print(">>>> Training started <<<<")
for key in classifiers:
    classifier = classifiers[key]
    scores = model_selection.cross_val_score(classifier, X_train, y_train, cv = FOLDS, scoring='accuracy')
    print("[%s] - accuracy: %0.2f " % (key, scores.mean()))
    classifier.fit(X_train, y_train)
    
    # Save classifier for prediction 
    classifiers[key] = classifier

# Predict & validation
preds = pd.DataFrame()

for key in classifiers:
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    preds[f"{key}"] = y_pred
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"{key} -> AUC: {auc:.3f}")

# RandomForest -> AUC: 0.937
# CatBoost -> AUC: 0.936
# LGBM -> AUC: 0.937
# ExtraTrees -> AUC: 0.937
# Stacked -> AUC: 0.938

preds[TARGET] = pd.DataFrame(y_test).reset_index(drop=True)

print(preds.head(10))   

test_preds = classifiers['Stacked'].predict_proba(test)[:,1]
threshold = pd.Series(test_preds).sort_values(ascending = False).head(34911).values[-1]
print(f"Current threshold is: {threshold}") #  0.2781261418060212
submission['submit_1'] = (test_preds > threshold).astype(int)
submission['submit_2'] = pd.read_csv("E:\\data\\kaggle_tabular\\submission\\submission_0430_2.csv")[TARGET]


submission[TARGET] = (submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis=1) >= 2).astype(int)
submission[['PassengerId', TARGET]].to_csv("E:\\data\\kaggle_tabular\\submission_0430_f3.csv", index = False)

# submission_0430_f3.csv 
# score 0.81722

import winsound as sd
def beepsound():
    fr = 800    # range : 37 ~ 32767
    du = 500     # 1000 ms ==1second
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

beepsound()
