# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:31 2019

@author: 033970
"""

import pandas as pd
import numpy as np
import pandas_profiling


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('D:/self-learn/kaggle/titanic/train.csv')
#pfr = pandas_profiling.ProfileReport(df_train)
#pfr.to_file("D:/self-learn/kaggle/titanic/example.html")


# Feature Engineering


df_train[['Pclass','Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False)
df_train[['Sex','Survived']].groupby('Sex').mean().sort_values(by='Survived', ascending=False)
df_train[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by='Survived', ascending=False)
df_train[['Parch','Survived']].groupby('Parch').mean().sort_values(by='Survived', ascending=False)
df_train[['Embarked','Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False)

df_train['isAlone'] = df_train['SibSp']+df_train['Parch']


def sex(x):
    if x=='female':
        return 1
    if x=='male':
        return 0
    
df_train['Sex'] = df_train['Sex'].apply(sex)


def embarked(x):
    if x=='C':
        return 2
    elif x=='Q':
        return 1
    elif x=='S':
        return 0
    else:
        np.nan

df_train['Embarked'] = df_train['Embarked'].apply(embarked)   

def isalone(x):
    if x>0:
        return 0
    elif x==0:
        return 1
    else:
        return np.nan

df_train['isAlone'] = df_train['isAlone'].apply(isalone)
df_train[['isAlone','Survived']].groupby('isAlone').mean().sort_values(by='Survived', ascending=False)

# Young:Age<=15,Old dosent matter
age_d1 = dict()
for i in range(10,80,10):
    age_d1[i] = df_train[(df_train['Age']<=i) &(df_train['Age']>i-10)][['Age','Survived']].mean()['Survived']
    
age_d2= dict()    
for i in range(80,50,-1):
    age_d2[i] = df_train[df_train['Age']>=i][['Age','Survived']].mean()['Survived']


def isteen(x):
    if x>15:
        return 0
    elif x<=15 and x>=0:
        return 1
    else:
        return np.nan  
    
df_train['isTeen'] = df_train['Age'].apply(isteen)
df_train[['isTeen','Survived']].groupby('isTeen').mean().sort_values(by='Survived', ascending=False)

df_train.dropna(axis=0,how='any',inplace=True)

# Fit the model
x_train,x_test, y_train, y_test =train_test_split(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone','isTeen']],df_train['Survived'],test_size=0.4,random_state=6)
rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
rf.score(x_test,y_test)










