# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:31 2019
@author: 033970
"""

import pandas as pd
import numpy as np
import pandas_profiling
import re

# visualization
import seaborn as sns
import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

global_path = 'D:/self-learn/kaggle/titanic'

df_train = pd.read_csv(global_path + '/train.csv')
#pfr = pandas_profiling.ProfileReport(df_train)
#pfr.to_file("D:/self-learn/kaggle/titanic/example.html")


# Feature Engineering


df_train[['Pclass','Survived']].groupby('Pclass').mean().sort_values(by='Survived', ascending=False)
df_train[['Sex','Survived']].groupby('Sex').mean().sort_values(by='Survived', ascending=False)
df_train[['SibSp','Survived']].groupby('SibSp').mean().sort_values(by='Survived', ascending=False)
df_train[['Parch','Survived']].groupby('Parch').mean().sort_values(by='Survived', ascending=False)
df_train[['Embarked','Survived']].groupby('Embarked').mean().sort_values(by='Survived', ascending=False)
df_train[['Fare','Survived']].groupby('Fare').mean().sort_values(by='Survived', ascending=False)





def feature_engineer(df_train):
    # Transform title feature
    def get_title(x):
        title_search = re.search('([A-Za-z]+)\.',x)
        if title_search:
            return title_search.group(1)
        return ''
    
    df_train['Title'] = df_train['Name'].apply(get_title)
    df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_train['Title'] = df_train['Title'].replace('Mlle', 'Miss')
    df_train['Title'] = df_train['Title'].replace('Ms', 'Miss')
    df_train['Title'] = df_train['Title'].replace('Mme', 'Mrs')
    title_mapping = {'Mr':1,'Master':2,'Miss':3,'Mrs':4,'Rare':5}
    df_train['Title'] = df_train['Title'].map(title_mapping)
    df_train['Title'] = df_train['Title'].fillna(0)
    
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
    
    df_train['Embarked'].fillna('S',inplace=True)
    df_train['Embarked'] = df_train['Embarked'].apply(embarked)   
    
    def isalone(x):
        if x>0:
            return 0
        elif x==0:
            return 1
        else:
            return np.nan
    
    df_train['isAlone'] = df_train['isAlone'].apply(isalone)
    
    def isteen(x):
        if x>15:
            return 0
        elif x<=15 and x>=0:
            return 1
        else:
            return np.nan  
        
    
    
    
#    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
    
    # Fill na in ages
    age_mean = df_train['Age'].mean()
    age_std = df_train['Age'].std()
    age_null_count = df_train['Age'].isnull().sum()
    age_null_random = np.random.randint(age_mean-age_std,age_mean+age_std,size=age_null_count)
    df_train['Age'][pd.isna(df_train['Age'])] = age_null_random
    df_train['Age'] = df_train['Age'].astype(int)
    # Transform age to categorical data
    df_train.loc[ df_train['Age'] <= 16, 'Age'] = 0
    df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1
    df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2
    df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3
    df_train.loc[ df_train['Age'] > 64, 'Age'] = 4
    
    
    df_train['isTeen'] = df_train['Age'].apply(isteen)
    
    # Fill na in fare
    df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())
    # Transform fare to categorical data
    df_train.loc[ df_train['Fare'] <= 7.91, 'Fare'] = 0
    df_train.loc[(df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.454), 'Fare'] = 1
    df_train.loc[(df_train['Fare'] > 14.454) & (df_train['Fare'] <= 31), 'Fare'] = 2
    df_train.loc[(df_train['Fare'] > 31) , 'Fare'] = 3
    df_train['Fare'] = df_train['Fare'].astype(int)
    
    
    df_train['Has_cabin'] = df_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
    
    df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)
    
    return df_train
    


df_train = feature_engineer(df_train)






#df_train.dropna(axis=0,how='any',inplace=True)
#df_train.fillna(-1,inplace=True)

# Fit the model
x_train,x_test, y_train, y_test =train_test_split(df_train,df_train['Survived'],test_size=0.4)
rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
rf.score(x_test,y_test)


y = df_train['Survived']
x = df_train.drop(['Survived'],axis=1)

rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
rf.fit(x,y)
rf.score(x,y)
scores = cross_val_score(rf, x, y, cv=5)
print(scores.mean())

svc = SVC()
svc.fit(x,y)
svc.score(x,y)
scores = cross_val_score(svc, x, y, cv=5)
print(scores.mean())

lr = LogisticRegression()
lr.fit(x,y)
lr.score(x,y)
scores = cross_val_score(lr, x, y, cv=5)
print(scores.mean())


kn = KNeighborsClassifier()
kn.fit(x,y)
kn.score(x,y)
scores = cross_val_score(kn, x, y, cv=5)
print(scores.mean())


rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
svc = SVC()
lr = LogisticRegression()
kn = KNeighborsClassifier()
voting_clf = VotingClassifier( estimators=[("lr", lr), ("rf", rf), ("svc", svc),('kn',kn)], voting="hard" )
voting_clf.fit(x,y)
voting_clf.score(x,y)
scores = cross_val_score(voting_clf, x, y, cv=5)
print(scores.mean())


# Get the test result
df_test = pd.read_csv(global_path + '/test.csv')
df_test = feature_engineer(df_test)
df_test.fillna(-1,inplace=True)
#test_predict = rf.predict(df_test)
test_predict = svc.predict(df_test)
#test_predict = voting_clf.predict(df_test)
df_res = pd.DataFrame(data={'PassengerId':df_test['PassengerId'],'Survived':test_predict})

df_res.to_csv(global_path + '/result00.csv',index=False)
