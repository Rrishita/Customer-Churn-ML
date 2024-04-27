#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[30]:


import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Reading the dataset

# In[10]:


df = pd.read_csv("telecom_churn.csv")
df.head()


# ## Number of rows and coulmns in the dataset

# In[12]:


df.shape


# ## Info of columns in the dataset

# In[13]:


df.info()


# ## Checking for null values

# In[14]:


df.isnull().sum()


# ## Data Exploration

# In[16]:


df['churn'].value_counts()


# In[18]:


483/3333 * 100 #Percentage of customers churned


# In[19]:


df['international plan'].value_counts()


# In[20]:


df['customer service calls'].value_counts()


# In[21]:


df.corr()


# In[22]:


sns.heatmap(df.corr())


# In[25]:


sns.jointplot(x='total intl charge', y='total intl minutes', data =df)


# In[ ]:


sns.pairplot(data=df)


# ## Data Transformation
# ### 1. Numerical Columns = 16
# ### 2. Categorical Columns = 5 (churn, voicemail plan, international plan, state, phone number)
# #ignoring state and phone number column as it as no impact on our data

# In[47]:


def churn(s):
    return 1 if s == True else 0
def voice_mail_plan(s):
    return 1 if s == 'yes' else 0
def international_plan(s):
    return 1 if s == 'yes' else 0


# In[50]:


df['churn'] = df['churn'].apply(churn)
df['voice mail plan'] = df['voice mail plan'].apply(voice_mail_plan)
df['international plan'] = df['international plan'].apply(international_plan)


# In[51]:


df.head()


# In[52]:


df['area code'].value_counts()


# In[31]:


df_dummy = pd.get_dummies(df, columns= ['area code'])
df_dummy.head()


# In[57]:


df_dummy.info()


# In[58]:


df_dummy.shape


# In[60]:


df_dummy.columns


# ## Independent(X) and dependent variables(Y)

# In[86]:


X = df_dummy[['account length', 'international plan',
       'voice mail plan', 'number vmail messages', 'total day minutes',
       'total day calls', 'total day charge', 'total eve minutes',
       'total eve calls', 'total eve charge', 'total night minutes',
       'total night calls', 'total night charge', 'total intl minutes',
       'total intl calls', 'total intl charge', 'customer service calls', 
       'area code_408', 'area code_415', 'area code_510']]
y = df_dummy[['churn']]


# ## Data split into train and test

# In[68]:


from sklearn.model_selection import train_test_split


# In[71]:


get_ipython().run_line_magic('pinfo', 'train_test_split')


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(  X, y, test_size=0.30, random_state=101)


# In[ ]:


'''X_train,  
y_train'''  -> to train the model
'''X_test,
   y_test''' -> to test the model


# ## Decision Tree
# 

# In[83]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[112]:


tree = DecisionTreeClassifier()
rfc = RandomForestClassifier()


# In[107]:


tree.fit(X_train, y_train)


# In[91]:


pred = tree.predict(X_test)


# In[93]:


pred.shape


# In[94]:


y_test.shape


# In[95]:


from sklearn.metrics import confusion_matrix, classification_report


# In[98]:


confusion_matrix(pred, y_test)


# In[102]:


print(classification_report(pred, y_test))


# ## Random Forest

# In[114]:


from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier()


# In[115]:


rfc.fit(X_train, y_train)


# In[118]:


pred_rfc = rfc.predict(X_test)


# In[119]:


confusion_matrix(pred_rfc, y_test)


# In[120]:


print(classification_report(pred_rfc, y_test))


# ### 85% accuracy with decision tree
# ### 92% accuracy with random forest

# In[ ]:




