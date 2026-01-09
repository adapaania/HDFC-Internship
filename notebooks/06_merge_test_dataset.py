#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Modeling – Feature Engineering & Modeling
# 
# This notebook is part of an end-to-end credit risk modeling project completed
# during my Data Science Internship at HDFC Capital Advisors Ltd.
# 
# ⚠️ Note: Due to data confidentiality, raw datasets are not included.
# The notebook demonstrates methodology, feature engineering logic,
# modeling approach, and evaluation techniques.
# 

# In[56]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


LOAD_DATA = False  # Set to True only in secure local environment

if LOAD_DATA:
    with open('./data/senior_ds_test/data/test/test_flag.csv', 'r') as f:
        df_flag = json.load(f)
else:
    df_flag = [] 


# In[58]:


df_flag.head()


# In[59]:


df_flag.columns


# ## Accounts and Enquiry Data

# In[60]:


df_acc = pd.read_csv("/Users/aaniaadap/Desktop/HDFC Internship/df_acc_test.csv")


# In[61]:


df_acc.shape


# In[62]:


df_enq = pd.read_csv("/Users/aaniaadap/Desktop/HDFC Internship/df_enq_test.csv")


# In[63]:


df_enq.shape


# In[64]:


df = pd.merge(
    pd.merge(df_flag, df_enq, on='uid', how='inner'),
    df_acc, on='uid', how='inner'
)


# In[65]:


df.shape


# In[66]:


columns = [
    'NAME_CONTRACT_TYPE',
    'accList_avg_overdues', 
    'accList_max_overdues', 
    'accList_min_overdues', 
    'accList_min_late_payments', 
    'accList_median_del_payments', 
    'accList_min_del_payments', 
    'accList_median_DPD', 
    'accList_min_DPD'
]

df = df.drop(columns = columns)


# In[67]:


df.shape


# In[68]:


df.columns.to_list()


# In[70]:


df.to_csv('df_test.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




