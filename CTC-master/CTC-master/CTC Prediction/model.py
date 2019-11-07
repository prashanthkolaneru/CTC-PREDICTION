#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


Dataset = pd.read_csv(r'C:/Users/User/payscale.csv')


# In[4]:


Dataset.head()


# In[5]:


X = Dataset.iloc[:,:-1].values


# In[6]:


y = Dataset.iloc[:,1].values


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel("WEX")
plt.ylabel("CTC")
plt.scatter(X,y,color='red',marker='+')


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)


# In[10]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[11]:


regressor.score(X_test,y_test)


# In[12]:


regressor.score(X_train,y_train)


# In[13]:


y = regressor.predict(X_test)


# In[14]:


df  = pd.DataFrame(X_test)
df["CTC "] = y
df["Actual CTC"] = y_test
df.columns = ["WEX" , "PREDICTED CTC", "ACTUAL CTC"]


# In[15]:


df


# In[16]:


from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y)))


# In[22]:


from sklearn.metrics import  r2_score
print(r2_score(y_test,y))


# In[23]:


regressor.intercept_


# In[24]:


regressor.coef_


# In[26]:


y = 10578+13711*4


# In[27]:


y


# In[30]:


pkl_filename = "LG.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(regressor, file)


# In[ ]:




