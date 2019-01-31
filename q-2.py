#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sklearn as sk
from sklearn.model_selection import train_test_split
eps=np.finfo(float).eps
from binarytree import tree,Node
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from operator import itemgetter
import copy
import collections
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


# In[70]:


names=['ID','Age','Experience','Income(in p.a)','Zip','FamilySize','Expenditure(p.m)','EduLevel','MortageValue','Personalloan','SecurityAcc','CD account','InternetBanking','CreditCard']
df=pd.read_csv("LoanDataset/data.csv",names=names)
# df


# In[71]:


numeric=['Age','Experience','Income(in p.a)','FamilySize','Expenditure(p.m)','EduLevel','MortageValue']
categoric=['SecurityAcc','CD account','InternetBanking','CreditCard']


# In[72]:


X =df.drop(['Personalloan','ID','Zip'],axis=1)
y=df['Personalloan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
df=pd.concat([X_train,y_train],axis=1)


# In[73]:


#Just seperation of class
def separateByClass(df):
#     print(dataset)
    separated = {}
    for index,row in df.iterrows():
        if  row['Personalloan'] in separated:
            separated[row['Personalloan']].append(list(row))
        else:
             separated[row['Personalloan']]=[]
    return separated


# In[74]:


def mean(numbers):
    return sum(numbers)/float(len(numbers))


# In[75]:


# sqrt((summa(x-avg)^2)/n)
def stdev(numbers):
    avg = mean(numbers)
    varianceNum=0.0
    for x in numbers:
        varianceNum=varianceNum+pow(x-avg,2)
    variance=varianceNum/float(len(numbers)-1)
#     variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


# In[76]:


def summarize(dataset):
    summaries={}
    df1=pd.DataFrame(dataset)
    names=['Age','Experience','Income(in p.a)','FamilySize','Expenditure(p.m)','EduLevel','MortageValue','SecurityAcc','CD account','InternetBanking','CreditCard','Personalloan']
    namesattr=['Age','Experience','Income(in p.a)','FamilySize','Expenditure(p.m)','EduLevel','MortageValue','SecurityAcc','CD account','InternetBanking','CreditCard']
    df1.columns=names
#     print(df1)
    df1=df1.drop(['Personalloan'],axis=1)
    for attribute in namesattr:
        summaries[attribute]=(mean(df1[attribute]), stdev(df1[attribute]))
#         summaries.append((mean(df1[attribute]), stdev(df1[attribute])))
    return summaries


# In[77]:


#har class ke har ek attribute ke liye mean and variance
def partitionByClass(df):
#     print(dataset)
    separated = separateByClass(df)
#     print("seperated")
#     print(seperated)
    summaries = {}
    #seperated by class list
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


# In[78]:


#calculate gaussian distribution
def Gaussian(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


# In[79]:


#calculate gaussian distribution of each attrbute in a given row
def getClassProbabilities(summaries,summaryCat,row):
    classCol= df.keys()[-1]
    probabilities = {}
    probabilitiescat={}
    for classValue, classSummaries in summaries.items():
        num = len([df[classCol]==classValue])
        den = len(df[classCol])
        fraction = num/(den+eps)
        probabilities[classValue] = 1
        probabilitiescat[classValue] =fraction
        flag=0
        for attr,meanvar in classSummaries.items():
            mean,stdev=meanvar
            x=row[attr]
            
            if attr in numeric:
                probabilities[classValue] *= Gaussian(x, mean, stdev)
            else:
#                 print(x)
                flag=1
                probabilitiescat[classValue]*=summaryCat[classValue][attr][x]
        if flag==1:
            probabilities[classValue]=probabilities[classValue]*probabilitiescat[classValue]
    return probabilities


# In[80]:


#most dominating attribute probablity
def getResultClass(summaries,summaryCat,row):
    probabilities = getClassProbabilities(summaries,summaryCat,row)
    if probabilities[0]>probabilities[1]:
        return 0
    else:
        return 1


# In[81]:


def predict(summaries,summaryCat,testSet):
    predictions = []
    for index,row in testSet.iterrows():
        result = getResultClass(summaries,summaryCat,row)
        #result is basically in format of 1/0
        predictions.append(result)
    return predictions


# In[82]:


# print(df)
def summaryCategory(df,attribute,value):
    cat={}
#     for attribute in categoric:
#     print(attribute)
    if df.empty==True:
        return
    classCol= df.keys()[-1] 
#         resultValues=df[classCol].unique() 
    attributeNames=df[attribute].unique()
    entropy2 = 0
    for attr in attributeNames:
        num = len(df[attribute][df[attribute]==attr][df[classCol]==value])
        den = len(df[attribute][df[attribute]==attr])
        fraction = num/(den+eps)
        cat[attr]=fraction
    return cat


# In[83]:


def summaryCategoryattr(df,classCol):
    cat={}
    for attribute in categoric:
        cat[attribute]=summaryCategory(df,attribute,classCol)
    return cat


# In[84]:


summaries = partitionByClass(df)
summaryCat={}

summaryCat[0]=summaryCategoryattr(df,0)
summaryCat[1]=summaryCategoryattr(df,1)


testfile=sys.argv[1]
names=['ID','Age','Experience','Income(in p.a)','Zip','FamilySize','Expenditure(p.m)','EduLevel','MortageValue','SecurityAcc','CD account','InternetBanking','CreditCard']
df2=pd.DataFrame()
try:
    df2=pd.read_csv(testfile,names=names)
    predictions = predict(summaries,summaryCat,df2)
    print("-------------------------------Loan Prediction------------------------------\n")
    print(predictions)
except:
    df2=X_test
    predictions = predict(summaries,summaryCat,df2)
    print(confusion_matrix(y_test, predictions))  
    print(classification_report(y_test,predictions)) 
    print("Accuracy: ",accuracy_score(y_test, predictions))


# In[85]:


print("----------------------------System Results----------------------------------")
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))

