import numpy as np 
import pandas as pd 

sub = pd.read_csv('E:\data\kaggle_tabular/sample_submission.csv')

sub1 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0429_voting8_1.csv') 
sub2 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0430_5.csv') 
sub3 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0430_3.csv') 
sub4 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0429_voting6.csv') 
sub5 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0430_f4.csv') 
sub6 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0430_2.csv') 
sub7 = pd.read_csv('E:\data\kaggle_tabular\submission\\submission_0429_voting9.csv') 


res = (sub1['Survived'] + sub2['Survived'] + sub3['Survived'] + sub4['Survived'] + sub5['Survived'] + sub6['Survived'] + sub7['Survived'])/7
sub.Survived = np.where(res > 0.5, 1, 0).astype(int)

sub.to_csv('E:\data\kaggle_tabular\submission\\submission_0430_last100.csv', index = False)
print(sub['Survived'].mean())

# submission_0430_last100.csv 
# score 0.81730