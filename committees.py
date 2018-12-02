#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from io import StringIO


# In[40]:


import seaborn as sns


# In[36]:


# For Naive Bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[39]:


# For model selection/benchmarking
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


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


# In[37]:


# Multinomial Naive Bayes Classifier
X_train, X_test, y_train, y_test = train_test_split(df['DEVICENAME'], df['REVIEWADVISECOMM'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[38]:


print(clf.predict(count_vect.transform(["Heart valve"])))


# In[42]:


# Model selection/benchmarking

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []

for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


# In[43]:


# Visualization
fig = plt.figure(figsize=(8,6))
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[44]:


cv_df.groupby('model_name').accuracy.mean()


# In[45]:


model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[48]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.REVIEWADVISECOMM.values, yticklabels=category_id_df.REVIEWADVISECOMM.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[51]:


# Chi Square for the Linear SVC model
model.fit(features, labels)
N = 2
for REVIEWADVISECOMM, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("# '{}':".format(REVIEWADVISECOMM))
  print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
  print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))


# In[52]:


# Classification report for each class
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names=df['REVIEWADVISECOMM'].unique()))

