# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:31:31 2019

@author: 033970
"""

import pandas as pd
import numpy as np
import pandas_profiling

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

df_train = pd.read_csv('E:/MAFN/Self-study/kaggle/kaggle-master/titanic/train.csv')
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
    
    def isteen(x):
        if x>15:
            return 0
        elif x<=15 and x>=0:
            return 1
        else:
            return np.nan  
        
    df_train['isTeen'] = df_train['Age'].apply(isteen)
    
    
    df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
    df_train.loc[ df_train['Age'] <= 16, 'Age'] = 0
    df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1
    df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2
    df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3
    df_train.loc[ df_train['Age'] > 64, 'Age']
    
    
    df_train['Fare'] = df_train['Fare'].fillna(df_train['Fare'].median())
    df_train.loc[ df_train['Fare'] <= 7.91, 'Fare'] = 0
    df_train.loc[(df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.454), 'Fare'] = 1
    df_train.loc[(df_train['Fare'] > 14.454) & (df_train['Fare'] <= 31), 'Fare']   = 2
    df_train.loc[ df_train['Fare'] > 31, 'Fare'] = 3
    
    return df_train
    

df_train = feature_engineer(df_train)


df_train[['isAlone','Survived']].groupby('isAlone').mean().sort_values(by='Survived', ascending=False)
# Young:Age<=15,Old dosent matter
age_d1 = dict()
for i in range(10,80,10):
    age_d1[i] = df_train[(df_train['Age']<=i) &(df_train['Age']>i-10)][['Age','Survived']].mean()['Survived']
    
age_d2= dict()    
for i in range(80,50,-1):
    age_d2[i] = df_train[df_train['Age']>=i][['Age','Survived']].mean()['Survived']





#df_train.dropna(axis=0,how='any',inplace=True)
df_train.fillna(-1,inplace=True)

# Fit the model
x_train,x_test, y_train, y_test =train_test_split(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'],test_size=0.4)
rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
rf.score(x_test,y_test)



rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
rf.fit(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
rf.score(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
scores = cross_val_score(rf, df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']], df_train['Survived'], cv=5)
print(scores.mean())

svc = SVC()
svc.fit(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
svc.score(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
scores = cross_val_score(svc, df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']], df_train['Survived'], cv=5)
print(scores.mean())

lr = LogisticRegression()
lr.fit(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
lr.score(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
scores = cross_val_score(lr, df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']], df_train['Survived'], cv=5)
print(scores.mean())


kn = KNeighborsClassifier()
kn.fit(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
kn.score(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
scores = cross_val_score(kn, df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']], df_train['Survived'], cv=5)
print(scores.mean())


rf = RandomForestClassifier(n_estimators=20,max_depth=10,min_samples_split=2)
svc = SVC()
lr = LogisticRegression()
kn = KNeighborsClassifier()
voting_clf = VotingClassifier( estimators=[("lr", lr), ("rf", rf), ("svc", svc),('kn',kn)], voting="hard" )
voting_clf.fit(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
voting_clf.score(df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']],df_train['Survived'])
scores = cross_val_score(voting_clf, df_train[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']], df_train['Survived'], cv=5)
print(scores.mean())


# Get the test result
df_test = pd.read_csv('E:/MAFN/Self-study/kaggle/kaggle-master/titanic/test.csv')
df_test = feature_engineer(df_test)
df_test.fillna(-1,inplace=True)
#test_predict = rf.predict(df_test[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']])
#test_predict = svc.predict(df_test[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']])
test_predict = voting_clf.predict(df_test[['Pclass','Sex','Age','SibSp','Parch','Embarked','isAlone']])
df_res = pd.DataFrame(data={'PassengerId':df_test['PassengerId'],'Survived':test_predict})

df_res.to_csv('E:/MAFN/Self-study/kaggle/kaggle-master/titanic/result00.csv',index=False)





