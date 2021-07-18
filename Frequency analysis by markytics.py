#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


# In[3]:


pd.set_option('display.max_columns',None)


# In[4]:


df=pd.read_csv('Frequency.csv',delimiter='\t')


# In[5]:


df


# In[6]:


df.shape


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.columns


# In[10]:


#Lets check the null value
df.isnull().sum()


# as we can see there is no misssing valuein the dataset

# In[11]:


df['Sales Value'].value_counts(normalize=True)


# Dropping some dupicate values from the dataset

# In[12]:


df.dropna(inplace=True)


# In[13]:


df.shape


# no duplicate values in the dataset

# In[14]:


df.head()


# In[15]:


#lets create heatmap on the dataset
sns.heatmap(df.isnull())


# In[ ]:


df


# In[16]:


#lets find the correlation coefficient values in the dataset

# Finds correlation between Independent and dependent attributes

plt.figure(figsize = (18,18))
sns.heatmap(df.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# In[17]:


df['frequency']=1


# In[18]:


df1=df.groupby('Outlet ID',as_index=False).agg({'frequency':'sum','Sales Value':'sum'})
df1.head()


# In[19]:


df1.groupby('frequency').agg({'Sales Value':'mean'}).plot()


# As we can see number of sales value with purchasiing is increaing with frequncy increaing.

# In[20]:


### Sales Value
df1[df1['frequency']==2]['Sales Value'].sum()


# In[21]:


## Count of number of outlets
df1[df1['frequency']==2].shape[0]


# In[22]:


### Sales Value
df1[df1['frequency']==1]['Sales Value'].sum()


# In[24]:


## Count of number of outlets
df1[df1['frequency']==1].shape[0]


# In[26]:


### Sales Value
df1[df1['frequency']==3]['Sales Value'].sum()


# In[28]:


## Count of number of outlets
df1[df1['frequency']==3].shape[0]


# In[29]:


### Sales Value
df1[df1['frequency']==4]['Sales Value'].sum()


# In[30]:


## Count of number of outlets
df1[df1['frequency']==4].shape[0]


# In[31]:


### Sales Value
df1[df1['frequency']==5]['Sales Value'].sum()


# In[32]:


## Count of number of outlets
df1[df1['frequency']==5].shape[0]


# In[33]:


### Sales Value
df1[df1['frequency']==6]['Sales Value'].sum()


# In[34]:


## Count of number of outlets
df1[df1['frequency']==6].shape[0]


# In[35]:


### Sales Value
df1[df1['frequency']==7]['Sales Value'].sum()


# In[36]:


## Count of number of outlets
df1[df1['frequency']==7].shape[0]


# In[37]:


### Sales Value
df1[df1['frequency']==8]['Sales Value'].sum()


# In[38]:


## Count of number of outlets
df1[df1['frequency']==8].shape[0]


# In[39]:


### Sales Value
df1[df1['frequency']==9]['Sales Value'].sum()


# In[40]:


## Count of number of outlets
df1[df1['frequency']==9].shape[0]


# In[ ]:




