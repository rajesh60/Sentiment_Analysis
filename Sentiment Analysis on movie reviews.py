#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_table("train.tsv")


# In[3]:


train_Y = train_data["Sentiment"]


# In[4]:


train_X = train_data["Phrase"]


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.20, random_state=42)


# In[6]:


# seriesToListX_Train = list(X_train)
# seriesToListX_test  = list(X_test)


# In[7]:


CountVector = CountVectorizer()


# In[8]:


Xlist_train = CountVector.fit_transform(X_train)
Xlist_test  = CountVector.transform(X_test)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier


# In[10]:


knn = KNeighborsClassifier(n_neighbors= 10)


# In[11]:


knn.fit(Xlist_train,y_train)


# In[ ]:


predict = knn.predict(Xlist_test)
from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:




