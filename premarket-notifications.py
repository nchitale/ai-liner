#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[3]:


path = '/Users/nandini/Desktop/dvhacks/'
devices = pd.read_csv('pmn96cur.txt', sep="\|", engine="python")


# In[7]:


list(devices)


# In[23]:


devices = devices[['KNUMBER', 'REVIEWADVISECOMM', 'DATERECEIVED', 'DECISIONDATE']]
devices


# In[21]:


date_received = pd.to_datetime(devices['DATERECEIVED'])
decision_date = pd.to_datetime(devices['DECISIONDATE'])


# In[151]:


days_to_decision = decision_date - date_received
days_to_decision = days_to_decision.iloc[0:76131]


# In[152]:


days_to_decision = days_to_decision.dt.days
days_to_decision


# In[153]:


year_received = date_received.dt.year
year_received = year_received.iloc[0:76131]
year_received = year_received - 1990
year_received


# In[154]:


X = year_received.values.reshape(-1,1)
y = days_to_decision


# In[155]:


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.4, random_state=101)


# In[156]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[157]:


predictions = lm.predict(X_test)


# In[159]:


plt.scatter(y_test, predictions)


# In[160]:


days_to_decision

