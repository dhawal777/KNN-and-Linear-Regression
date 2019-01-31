#!/usr/bin/env python
# coding: utf-8

# In[110]:


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
import sys


# In[111]:


def iris(df2):
    def euclidean_distance(x, y): 
#row by row i.e all features resultant euclid distance in single line
        return np.sqrt(np.sum((x - y) ** 2))


    names=['sepal-length','sepal-width','petal-length','petal-width','Class']
    df=pd.read_csv("Iris/Iris.csv",names=names)

    X =df.drop(['Class'],axis=1)
    y=df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df=X_train
    y_train=list(y_train)

    def predict(Xtest,k):
        y_res=[]
        for index,row in Xtest.iterrows():
            result=[]
            for index1,row1 in df.iterrows():
                    result.append(euclidean_distance(row,row1))
            df1=pd.DataFrame(
            {
                'dist':result,
                'class':y_train
            })
            df1=df1.sort_values(by=['dist'])
            count=0;
    #         k=5
            classVote={}
            max1=0
            res=""
            for index1,row1 in df1.iterrows():
                capital=row1['class']
                count=count+1
                if  capital in classVote:
                    classVote[capital] = classVote[capital]+1
                else:
                    classVote[capital]=1
                if classVote[capital]>=max1:
                    res=capital
                    max1=classVote[capital]
                if count==k:
                    break;
            y_res.append(res)
        return y_res
    if df2.empty:
        
        y_res=predict(X_test,5)
        print("------Confusion Matrix---------")
        print(confusion_matrix(y_test, y_res))  
        print(classification_report(y_test, y_res)) 
        print("Accuracy: ",accuracy_score(y_test,y_res))
    else:
#         X_test =df2.drop(['Class'],axis=1)
#         y_test=df2['Class']
        y_res=predict(df2,5)
        print("-------------------Predicted Results of Iris-------------------------------\n")
        print(y_res)
#         print("------Confusion Matrix---------")
#         print(confusion_matrix(y_test, y_res))  
#         print(classification_report(y_test, y_res)) 
#         print("Accuracy: ",accuracy_score(y_test,y_res))
        

    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    # knn.score(X_train, y_test)
    y_pred =knn.predict(X_test)  
    print("------System confusion Matrix---------")
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    print()
    print("System Accuracy: ",accuracy_score(y_test,y_pred))


# In[112]:


def Robot1(df2):
    def cosine_similarity(x, y):
        return 1-np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

    df = pd.read_table("RobotDataset/Robot1", delim_whitespace=True, names=('Class','A', 'B', 'C','D','E','F','ID'))
    X =df.drop(['Class','ID'],axis=1)
    y=df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df=X_train
    y_train=list(y_train)

    def predict(Xtest,k):
        y_res=[]
        for index,row in Xtest.iterrows():
            result=[]
            for index1,row1 in df.iterrows():
                    result.append(cosine_similarity(row,row1))
            df1=pd.DataFrame(
            {
                'dist':result,
                'class':y_train
            })
            df1=df1.sort_values(by=['dist'])
            count=0;
            classVote={}
            max1=0
            res=""
            for index1,row1 in df1.iterrows():
                capital=row1['class']
                count=count+1
                if  capital in classVote:
                    classVote[capital] = classVote[capital]+1
                else:
                    classVote[capital]=1
                if classVote[capital]>=max1:
                    res=capital
                    max1=classVote[capital]
                if count==k:
                    break;
            y_res.append(res)
        return y_res
    if df2.empty:
        y_res=predict(X_test,5)
        print(confusion_matrix(y_test, y_res))  
        print(classification_report(y_test, y_res)) 
        print(accuracy_score(y_test,y_res))
        
    else:    
        y_res=predict(df2,5)
        print("-----------------Predicted Output of Robot1-----------------------\n")
        print(y_res)
    print("SYSTEM ACCURACY AND CONFUSION MATRIX")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    # knn.score(X_train, y_test)
    y_pred =knn.predict(X_test)  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    print(accuracy_score(y_test,y_pred))


# In[113]:


def Robot2(df3):
    def cosine_similarity(x, y):
        return 1-np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

    df = pd.read_table("RobotDataset/Robot2", delim_whitespace=True, names=('Class','A', 'B', 'C','D','E','F','ID'))

    X =df.drop(['Class','ID'],axis=1)
    y=df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    df=X_train
    y_train=list(y_train)

    def predict(Xtest,k):
        y_res=[]
        for index,row in Xtest.iterrows():
            result=[]
            for index1,row1 in df.iterrows():
                    result.append(cosine_similarity(row,row1))
            df1=pd.DataFrame(
            {
                'dist':result,
                'class':y_train
            })
            df1=df1.sort_values(by=['dist'])
            count=0;
    #         k=5
            classVote={}
            max1=0
            res=""
            for index1,row1 in df1.iterrows():
                capital=row1['class']
                count=count+1
                if  capital in classVote:
                    classVote[capital] = classVote[capital]+1
                else:
                    classVote[capital]=1
                if classVote[capital]>=max1:
                    res=capital
                    max1=classVote[capital]
                if count==k:
                    break;
            y_res.append(res)
        return y_res
#     print("Dataframe",df3)
    if df3.empty:
        y_res=predict(X_test,5)
        print(confusion_matrix(y_test, y_res))  
        print(classification_report(y_test, y_res)) 
        print(accuracy_score(y_test,y_res))
    else:
        y_res=predict(df3,5)
        print("-----------------Predicted Output of Robot2-----------------------\n")
        print(y_res)
        
    print("SYSTEM ACCURACY AND CONFUSION MATRIX")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    # knn.score(X_train, y_test)
    y_pred =knn.predict(X_test)  
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    print(accuracy_score(y_test,y_pred))


# In[114]:


# python/python3 <file_name.py> <absolute_path_of_test_file_for that_subpart>
# testfile=sys.argv[1]
# df= pd.read_table(testfile, delim_whitespace=True, names=('Class','A', 'B', 'C','D','E','F','ID'))

print("--------------------------ROBOT 1-----------------------------")

df2=pd.DataFrame()
try:
    testfile=sys.argv[1]
    df2 = pd.read_table(testfile, delim_whitespace=True, names=('A', 'B', 'C','D','E','F','ID'))
    df2 =df2.drop(['ID'],axis=1)
#     df2=pd.read_csv(testfile,names=names)
    Robot1(df2)
except:
    Robot1(df2)


# In[115]:


print("--------------------------ROBOT 2-----------------------------")

df3=pd.DataFrame()
# print(df2)
try:
    testfile=sys.argv[2]
    df3 = pd.read_table(testfile, delim_whitespace=True, names=('A', 'B', 'C','D','E','F','ID'))
    df3 =df3.drop(['ID'],axis=1)
#     df2=pd.read_csv(testfile,names=names)
    Robot2(df3)
except:
    df3=df3.iloc[0:0]
    Robot2(df3)


# In[116]:


print("---------------------IRIS---------------------------------")
# iris()

names=['sepal-length','sepal-width','petal-length','petal-width']
df4=pd.DataFrame()
try:
    testfile=sys.argv[3]
    df4=pd.read_csv(testfile,names=names)
    iris(df4)
except:
    df4=df4.iloc[0:0]
    iris(df4)
    


