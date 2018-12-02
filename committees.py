#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import StringIO


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[33]:


from sklearn.feature_selection import chi2


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


# In[27]:


# has unique product codes
df = combined_df.drop_duplicates(subset='DEVICENAME')
df


# In[29]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')


# In[30]:


features = tfidf.fit_transform(df.DEVICENAME).toarray()


# In[31]:


labels = df.category_id


# In[32]:


features.shape 
# Each of 2796 product codes is represented by 9675 features
# Representing the tf-idf score for different unigrams and bigrams


# In[35]:


# Find terms most correlated with each advisory committee
N = 2
for REVIEWADVISECOMM, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(REVIEWADVISECOMM))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

