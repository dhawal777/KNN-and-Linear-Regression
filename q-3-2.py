#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
# from sklearn.utils import check_arrays


# In[32]:


df=pd.read_csv("AdmissionDataset/data.csv")
df
X =df.drop(['Chance of Admit ','Serial No.'],axis=1)
y=df['Chance of Admit ']


# *Standardizing consists in subtracting the mean and dividing by the standard deviation.*
# *The convention that you standardize predictions primarily exists so that the units of the regression coefficients are the same.*
# *when value are largely diffrent one is population and some other attribute is fraction*

# In[33]:


X = (X - X.mean())/X.std()#normalize the data(z-mu)/(sigma)
# y=(y-y.mean())/y.std()
#kyuki actual y ke paas pahuchna hai isliye ,jhe y ko normalize karne ka koi reason nahi lag raha


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
features=['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']
my_data=pd.concat([X_train,y_train],axis=1)


# In[35]:


# X = my_data.iloc[:,0:7]


# In[36]:


# X


# In[37]:


X=X_train


# In[38]:


ones = np.ones([X.shape[0],1])


# In[39]:


X = np.concatenate((ones,X),axis=1)


# In[40]:


#print(type(my_data))
#print(type(y_train))


# In[41]:


# y = my_data.iloc[:,7:8].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray


# *values converts it from pandas.core.frame.DataFrame to numpy.ndarray*

# In[42]:


y
y=pd.DataFrame(y_train)
y=y.values
# y


# *basically assumed m=0 and c=0 for each independent variable intially*

# In[43]:


theta = np.zeros([1,8])

# theta


# *learning rate needed along with direction*

# In[44]:


#set hyper parameters
alpha = 0.01


# *no of times we have to iterate to minimize rms*

# In[45]:


iters = 1000


# In[46]:


# def computeCost(X,y,theta):
#     tobesummed = np.power(((X @ theta.T)-y),2)
#     return np.sum(tobesummed)/(2 * len(X))


# *Calculating Gradient desent for MAE for predicting best theta values*

# In[47]:


def gradientDescentMeanAbsolute(X,y,theta,iters,alpha):
#     cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * ((X @ theta.T - y)/abs((X @ theta.T - y))), axis=0)
#         cost[i] = computeCost(X, y, theta)
    
    return theta


# *Calculating Gradient desent for MSE for predicting best theta values*

# In[48]:


def gradientDescent(X,y,theta,iters,alpha):
#     cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)
#         cost[i] = computeCost(X, y, theta)
    
    return theta

g = gradientDescent(X,y,theta,iters,alpha)
g1=gradientDescentMeanAbsolute(X,y,theta,iters,alpha)


# *Calculated Theta List*

# In[49]:


betaList=g[0]
betaList1=g1[0]
# betaList


# In[50]:


# fig, ax = plt.subplots()  
# ax.plot(np.arange(iters), cost, 'r')  
# ax.set_xlabel('Iterations')  
# ax.set_ylabel('Cost')  
# ax.set_title('Error vs. Training Epoch')  


# In[51]:


# c


# In[52]:


y_pred=[]
for index,row in X_test.iterrows():
    row=list(row)
    y1=0
    for i in range(1,8):
        y1=y1+betaList[i]*row[i-1]
    y1=y1+betaList[0]
    y_pred.append(y1)


# In[53]:


# y_pred


# **CALCULATING R2 SCORE FOR MSE**

# In[54]:


print("--------------------------r2 score MSE---------------------------------")
print("Score: ",r2_score(list(y_test),y_pred))


# In[55]:


def meanSquareError(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.square((y_true - y_pred)))


# In[56]:


print("----------------------------MSE---------------------------------")
print("Error: ",mean_squared_error(list(y_test),y_pred))
print("Error: ",meanSquareError(list(y_test),y_pred))


# **CALCULATING Error FOR MAPE**

# In[57]:


def meanAbsolutePercentageError(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[58]:


print("---------------------------Error MAPE--------------------------------")
print("Error: ",meanAbsolutePercentageError(list(y_test),y_pred))
# print("Score: ",r2_score(list(y_test),y_pred))


# In[59]:


def meanAbsoluteError(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))


# **CALCULATING MAE and Its R2_SCORE**

# In[60]:


print("---------------------------Error MAE--------------------------------")
print("Error: ",mean_absolute_error(list(y_test),y_pred))
print("Error: ",meanAbsoluteError(list(y_test),y_pred))
y_pred=[]
for index,row in X_test.iterrows():
    row=list(row)
    y1=0
    for i in range(1,8):
        y1=y1+betaList1[i]*row[i-1]
    y1=y1+betaList1[0]
    y_pred.append(y1)
    
print("---------------------------R2 Score--------------------------------")
print("Score: ",r2_score(list(y_test),y_pred))


# In[61]:


# from sklearn.linear_model import LinearRegression
# regressor=LinearRegression()
# regressor.fit(X_train,y_train)
# y_pred = regressor.predict(X_test) 
# print("--------------------------System r2 Score-------------------------------")
# print("System Score: ",r2_score(y_test,y_pred))

