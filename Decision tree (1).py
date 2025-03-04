#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[ ]:


data = pd.read_csv("iris.csv")


# In[ ]:


data


# In[ ]:


import seaborn as sns
counts = data["variety"].value_counts()
sns.barplot(data = counts)


# In[ ]:


data.info()


# In[ ]:


data[data.duplicated(keep=False)]


# In[ ]:


data[data.duplicated()]


# ### Observations
# - There are 150 rows and 5 columns
# - There are no null values 
# - There are 1 duplicated value 
# -The x-columns are  sepal.length,sepal.width,petal.length,petal.width
# - The y-column is variety
# - All x values are continous
# - y column is catgorical
# - There are three flower categories

# In[ ]:


data.drop_duplicates(keep='first', inplace = True)


# In[ ]:


data[data.duplicated()]


# In[ ]:


data = data.reset_index(drop=True)


# In[ ]:


data


# In[ ]:


labelencoder = LabelEncoder()
data.iloc[:,-1] = labelencoder.fit_transform(data.iloc[:,-1])
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.info()       


# In[ ]:


data['variety'] = pd.to_numeric(labelencoder.fit_transform(data['variety']))


# In[ ]:


data.info()


# In[ ]:


data.head(4)


# In[ ]:


X = data.iloc[:,0:4]
Y = data['variety']


# In[ ]:


X


# In[ ]:


Y


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
x_train


# In[ ]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=None)
model.fit(x_train,y_train)


# In[ ]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[ ]:




