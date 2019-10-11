# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:31 2019
@author: 033970
"""

import pandas as pd
import numpy as np
import pandas_profiling
import re

import xgboost as xgb
import seaborn as sns

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
from sklearn.model_selection import KFold

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
    df_train.loc[df_train['Age'][pd.isna(df_train['Age'])].index,'Age'] = age_null_random
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
df_test = pd.read_csv(global_path + '/test.csv')
PassengerId = df_test['PassengerId']
df_test = feature_engineer(df_test)


# set parameters to kfolds
ntrain = df_train.shape[0]
ntest = df_test.shape[0]
seed = 0
nfolds = 5
kf = KFold(nfolds,random_state=seed)

# Class to extend sklearn classifier
class SklearnHelper(object):
    def __init__(self,clf,params=None):
        if params == None:
            params = {}
        
        self.clf = clf(**params)
    
    def train(self,x_train,y_train):
        self.clf.fit(x_train,y_train)
    
    def predict(self,x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        self.clf.fit(x,y)
        
    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
    
    
# fit the model and return the predictions for train and test
def get_oof(clf,kf,nfolds,x_train,y_train,x_test):
    oof_train = np.zeros((x_train.shape[0],))
    oof_test = np.zeros((x_test.shape[0],))
    oof_test_skf = np.empty((nfolds,x_test.shape[0]))
    
    i = 0
    for train_index,test_index in kf.split(x_train):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        
        clf.train(x_tr,y_tr)
        
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)
        i+=1
    
    oof_test[:] = np.array([i for i in map(lambda x:1 if x>=0.5 else 0,oof_test_skf.mean(axis=0))])
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
    

# Creat np arrays for train,test and target
y_train =df_train['Survived'].ravel()
df_train = df_train.drop(['Survived'],axis=1)
x_train = df_train.values
x_test = df_test.values 

# Create different classifiers
rf = SklearnHelper(RandomForestClassifier,params={'n_estimators':20,'max_depth':10,'min_samples_split':2})
svc = SklearnHelper(SVC)
lr = SklearnHelper(LogisticRegression)
kn = SklearnHelper(KNeighborsClassifier)

# Create our OOF train and test predictions. These base results will be used as new features
rf_oof_train, rf_oof_test = get_oof(rf, kf,nfolds,x_train, y_train, x_test)
svc_oof_train, svc_oof_test = get_oof(svc, kf,nfolds,x_train, y_train, x_test)
lr_oof_train, lr_oof_test = get_oof(lr, kf,nfolds, x_train, y_train, x_test)
kn_oof_train, kn_oof_test = get_oof(kn, kf,nfolds, x_train, y_train, x_test)


x_train_2 = np.concatenate(( rf_oof_train, svc_oof_train, lr_oof_train, kn_oof_train), axis=1)
x_test_2 = np.concatenate(( rf_oof_test, svc_oof_test, lr_oof_test, kn_oof_test), axis=1)

gbm = xgb.XGBClassifier(
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train_2, y_train)
predictions = gbm.predict(x_test_2)


# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'PassengerId': PassengerId,'Survived': predictions })
StackingSubmission.to_csv("StackingSubmission.csv", index=False)

'''
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
'''