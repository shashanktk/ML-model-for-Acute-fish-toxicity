#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn import preprocessing
import pickle


# In[3]:


data=pd.read_excel('D:\Academics\Summer sem 2021\Machine learning\Train.xlsx') #Need to change the directory


# In[4]:


le = preprocessing.LabelEncoder()
column_1=pd.DataFrame(le.fit_transform(data.iloc[:,0]))
column_2=pd.DataFrame(le.fit_transform(data.iloc[:,1]))
X=pd.concat([column_1,column_2],axis=1)


# In[5]:


data_X=pd.concat([X,data.iloc[:,3:]], axis=1)


# In[6]:


data_Y=data.iloc[:,2]


# In[7]:


reg = LinearRegression().fit(data_X, data_Y)


# In[8]:


filename = 'Shashank_TumkurKarnick_229777'
pickle.dump(reg, open(filename, 'wb'))

