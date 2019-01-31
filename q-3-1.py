#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sys


# In[131]:


df=pd.read_csv("AdmissionDataset/data.csv")
df
X =df.drop(['Chance of Admit ','Serial No.'],axis=1)
y=df['Chance of Admit ']


# *Standardizing consists in subtracting the mean and dividing by the standard deviation.*
# *The convention that you standardize predictions primarily exists so that the units of the regression coefficients are the same.*
# *when value are largely diffrent one is population and some other attribute is fraction*

# In[132]:


X = (X - X.mean())/X.std()#normalize the data(z-mu)/(sigma)
# y=(y-y.mean())/y.std()
#kyuki actual y ke paas pahuchna hai isliye ,jhe y ko normalize karne ka koi reason nahi lag raha


# In[133]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
features=['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']
my_data=pd.concat([X_train,y_train],axis=1)


# In[134]:


X=X_train


# In[135]:


ones = np.ones([X.shape[0],1])


# In[136]:


X = np.concatenate((ones,X),axis=1)


# *values converts it from pandas.core.frame.DataFrame to numpy.ndarray*

# In[137]:


y
y=pd.DataFrame(y_train)
y=y.values


# *basically assumed m=0 and c=0 for each independent variable intially*

# In[138]:


theta = np.zeros([1,8])

# theta


# *learning rate needed along with direction*

# In[139]:


#set hyper parameters
alpha = 0.01


# *no of times we have to iterate to minimize rms*

# In[140]:


iters = 1000


# *Calculating gradient descent by 1/m*sum(x*thetaT-y)*

# In[141]:


def gradientDescent(X,y,theta,iters,alpha):
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
    return theta
g = gradientDescent(X,y,theta,iters,alpha)


# In[142]:


betaList=g[0]


# In[143]:


def predict(Xtest):
    y_pred=[]
    for index,row in Xtest.iterrows():
        row=list(row)
        y1=0
        for i in range(1,8):
            y1=y1+betaList[i]*row[i-1]
        y1=y1+betaList[0]
        y_pred.append(y1)
    return y_pred


# In[144]:


testfile=sys.argv[1]
df2=pd.DataFrame()
try:
    df2=pd.read_csv(testfile)
    df2 =df2.drop(['Serial No.'],axis=1)
    df2 = (df2 - df2.mean())/df2.std()
    y_pred=predict(df2)
    print("----------------------Output of Testing Data----------------------------------\n")
    print(y_pred)
except:
    df2=X_test
    y_pred=predict(df2)
    print("-------------------Linear regression R2 Score-----------------------------------\n")
    print("Score: ",r2_score(list(y_test),y_pred))
    


# In[145]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test) 
r2_score(y_test,y_pred)

