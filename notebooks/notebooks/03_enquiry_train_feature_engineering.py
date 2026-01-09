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

# In[38]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[39]:


pd.set_option('display.max_columns', 100)


# In[40]:


LOAD_DATA = False  # Set to True only in secure local environment

if LOAD_DATA:
    with open('./data/senior_ds_test/data/train/enquiry_data_train.json', 'r') as f:
        df = json.load(f)
else:
    df = []
    
flat_list = [item for sublist in df for item in sublist]
df_enq = pd.DataFrame(flat_list)


# In[41]:


df_enq.head()


# In[42]:


df_enq.columns


# In[43]:


df_enq.info()


# In[44]:


df_enq.shape


# In[45]:


df_enq["enquiry_date"] = pd.to_datetime(df_enq["enquiry_date"],format= "%Y-%m-%d")


# In[46]:


df_enq["uid"].value_counts().to_frame().sort_values(by='uid')


# In[47]:


df_enq["enquiry_type"].nunique()


# In[48]:


df_enq.loc[df_enq.duplicated(subset=["uid","enquiry_amt","enquiry_date"])].sort_values(by="enquiry_date")
       


# In[49]:


#No. of enquiries made by each customer for different categories. 
enq_counts = df_enq.groupby(["uid","enquiry_type"])["enquiry_type"].size().to_frame()

enq_counts.head(40)


# In[50]:


df_enq.shape


# ## Features

# In[51]:


df_enq['enquiry_year']=df_enq["enquiry_date"].dt.year

#month enquired
df_enq['enquiry_month']=df_enq["enquiry_date"].dt.month

# day enquired
df_enq['enquiry_day']=df_enq["enquiry_date"].dt.day


# In[52]:


df_enq["Avg_amount_enquired"] = df_enq["uid"].map(df_enq.groupby('uid')['enquiry_amt'].mean().round(2))
df_enq["max_amount_enquired"] = df_enq["uid"].map(df_enq.groupby('uid')['enquiry_amt'].max())
df_enq["min_amount_enquired"] = df_enq["uid"].map(df_enq.groupby('uid')['enquiry_amt'].min())


# In[53]:


df_enq['total_enquiries_per_customer'] = df_enq.groupby('uid')['enquiry_type'].transform('count')


# In[25]:


df_enq.head(10)


# In[54]:


# count of enquiry types per customer.
df_enq['unique_enquiry_types_per_customer'] = df_enq.groupby('uid')['enquiry_type'].transform('nunique')


# In[55]:


df_enq.shape


# In[56]:


df_enq[df_enq['uid'] == 'AAA09044550'][['enquiry_type', 'uid']].value_counts()


# In[57]:


df_enq['total_enquiries_per_type'] = df_enq.groupby(['enquiry_type'])['enquiry_type'].transform('count')


# In[58]:


#Percentage of each type of enquiries per customer by the total no of enquiries for that type

df_enq['enquiry_percentage'] = (df_enq['total_enquiries_per_customer'] / df_enq['total_enquiries_per_type']) * 100


# In[59]:


grouped = df_enq.groupby(['uid', 'enquiry_type']).size().reset_index(name='count')

# Pivot the DataFrame
pivot_table = grouped.pivot(index='uid', columns='enquiry_type', values='count').fillna(0).astype(int)

# Flatten the pivot table columns
pivot_table.columns = [f'{col}_total_enquiries' for col in pivot_table.columns]

# Reset index to prepare for merge
pivot_table.reset_index(inplace=True)

# Merge the pivoted data back into the main DataFrame (df_enq)
df_enq = pd.merge(df_enq, pivot_table, on='uid', how='left')


# In[61]:


enquiry_percentage_aggregates = df_enq.groupby('enquiry_type')['enquiry_percentage'].agg(['mean', 'sum', 'min', 'max']).reset_index()
enquiry_percentage_aggregates.columns = ['enquiry_type', 'mean_enquiry_percentage', 'total_enquiry_percentage', 'min_enquiry_percentage', 'max_enquiry_percentage']

# Merge aggregated statistics back into df_enq
df_enq = df_enq.merge(enquiry_percentage_aggregates, on='enquiry_type', how='left')


# In[62]:


df_enq.shape


# In[34]:


df_enq.head()


# In[63]:


# first enquiry date per type
df_enq['first_enquiry_date'] = df_enq.groupby(['uid', 'enquiry_type'])['enquiry_date'].transform('min')


# In[64]:


# last enquiry date per type
df_enq['last_enquiry_date'] = df_enq.groupby(['uid', 'enquiry_type'])['enquiry_date'].transform('max')


# In[65]:


max_date = df_enq['enquiry_date'].max()


# In[66]:


max_date


# In[69]:


df_enq.shape


# ###### COUNT OF ENQUIRIES ACC TO NO. OF MONTHS

# In[68]:


#count of enquiries in the last 1 month
a_month_ago = max_date - pd.DateOffset(months=1)
df_enq['enquiries_last_1_month'] = df_enq.groupby('uid')['enquiry_date'].transform(lambda x: x.gt(a_month_ago).sum())


# In[70]:


#count of enquiries in the last 3 months
three_months_ago = max_date - pd.DateOffset(months=3)
df_enq['enquiries_last_3_months'] = df_enq.groupby('uid')['enquiry_date'].transform(lambda x: x.gt(three_months_ago).sum())


# In[71]:


#count of enquiries in the last 6 months
six_months_ago = max_date - pd.DateOffset(months=6)
df_enq['enquiries_last_6_months'] = df_enq.groupby('uid')['enquiry_date'].transform(lambda x: x.gt(six_months_ago).sum())


# In[72]:


#count of enquiries in the last 9 months
nine_months_ago = max_date - pd.DateOffset(months=9)
df_enq['enquiries_last_9_months'] = df_enq.groupby('uid')['enquiry_date'].transform(lambda x: x.gt(nine_months_ago).sum())


# In[75]:


#count of enquiries in the last year 
max_date = df_enq['enquiry_date'].max()
one_year_ago = max_date - pd.DateOffset(years=1)
df_enq['enquiries_last_year'] = df_enq.groupby('uid')['enquiry_date'].transform(lambda x: x.gt(one_year_ago).sum())


# In[76]:


df_enq.head()


# In[78]:


df_enq.shape


# ###### ENQUIRY AMOUNT

# In[79]:


pivot_mean = pd.pivot_table(df_enq, 
                            index='uid', 
                            columns='enquiry_type', 
                            values='enquiry_amt', 
                            aggfunc='mean', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_mean.columns = [f'{credit}_mean_enquiry_amt' for credit in pivot_mean.columns]

# Reset the index to merge with the original dataframe
pivot_mean.reset_index(inplace=True)

df_enq = df_enq.merge(pivot_mean, on='uid', how='left')


# In[84]:


df_enq.shape


# In[81]:


pivot_median = pd.pivot_table(df_enq, 
                            index='uid', 
                            columns='enquiry_type', 
                            values='enquiry_amt', 
                            aggfunc='median', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_median.columns = [f'{credit}_median_enquiry_amt' for credit in pivot_median.columns]

# Reset the index to merge with the original dataframe
pivot_median.reset_index(inplace=True)

df_enq = df_enq.merge(pivot_median, on='uid', how='left')


# In[83]:


pivot_sum = pd.pivot_table(df_enq, 
                            index='uid', 
                            columns='enquiry_type', 
                            values='enquiry_amt', 
                            aggfunc='sum', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_sum.columns = [f'{credit}_total_enquiry_amt' for credit in pivot_sum.columns]

# Reset the index to merge with the original dataframe
pivot_sum.reset_index(inplace=True)

df_enq = df_enq.merge(pivot_sum, on='uid', how='left')


# In[106]:


df_enq.head()


# In[ ]:


df_enq.loc[df_enq['uid'] == 'AAA08065248', ['enquiry_type', 'total_enquiries_per_type', 'uid']]


# In[85]:


df_enq.shape


# In[86]:


def get_user_info(df, user_id):
    user_df = df[df['uid'] == user_id]
    return user_df.to_dict(orient='records')

# Get the information for user ID 'AAA09044550'
user_info = get_user_info(df, 'AAA08065248')

user_info


# ### FEATURE DROPS

# In[43]:


df_enq.loc[1:3]


# In[87]:


columns_to_drop_enq =[
    "enquiry_type",
    "enquiry_amt",
    "enquiry_date",
    "enquiry_year",
    "enquiry_month",
    "enquiry_day",
    "total_enquiries_per_type",
    "enquiry_percentage",
    "mean_enquiry_percentage",
    "total_enquiry_percentage",
    "min_enquiry_percentage",
    "max_enquiry_percentage",
    "first_enquiry_date",
    "last_enquiry_date"   
]
df_enq = df_enq.drop(columns=columns_to_drop_enq)


# In[88]:


df_enq.columns.to_list()


# In[89]:


df_enq.shape


# In[90]:


df_enq["uid"].nunique()


# In[91]:


df_enq["uid"].value_counts()


# In[92]:


df_enq.duplicated(subset="uid")


# In[93]:


df_enq = df_enq.drop_duplicates(subset="uid",keep="first")


# In[ ]:


df_enq.shape


# In[94]:


enq_prefix = 'enqList_'

# Add prefix to all columns in df_enq except 'uid'
df_enq = df_enq.rename(columns=lambda x: enq_prefix + x if x != 'uid' else x)


# In[95]:


df_enq.to_csv('df_enq.csv', index=False)


# In[96]:


user_dict = df_enq[df_enq["uid"]=='AAA09044550'].to_dict('records')


# In[97]:


user_dict


# In[55]:


df_enq.columns.to_list()


# In[ ]:




