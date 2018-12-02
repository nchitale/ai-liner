#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import StringIO


# In[4]:


path = '/Users/nandini/Desktop/dvhacks/'
devices = pd.read_csv('pmn96cur.txt', sep='\|', engine='python')
product_codes = pd.read_csv('foiclass.txt', sep='\|', engine='python')


# In[3]:


list(devices)


# In[6]:


list(product_codes)


# In[7]:


product_codes = product_codes[['PRODUCTCODE', 'DEVICENAME']]
product_codes


# In[5]:


devices = devices[['REVIEWADVISECOMM', 'PRODUCTCODE']]
devices


# In[10]:


combined_df = pd.merge(devices, product_codes, on='PRODUCTCODE')
combined_df


# In[14]:


col = ['REVIEWADVISECOMM', 'DEVICENAME']
combined_df = combined_df[col]
combined_df


# In[16]:


combined_df.columns = ['REVIEWADVISECOMM', 'DEVICENAME']
combined_df['category_id'] = combined_df['REVIEWADVISECOMM'].factorize()[0]
combined_df


# In[17]:


category_id_df = combined_df[['REVIEWADVISECOMM', 'category_id']].drop_duplicates().sort_values('category_id')
category_id_df


# In[18]:


category_to_id = dict(category_id_df.values)
category_to_id


# In[19]:


id_to_category = dict(category_id_df[['category_id', 'REVIEWADVISECOMM']].values)
id_to_category


# In[21]:


fig = plt.figure(figsize=(8,6))
combined_df.groupby('REVIEWADVISECOMM').DEVICENAME.count().plot.bar(ylim=0)
plt.show()

