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

# In[21]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


import warnings
warnings.filterwarnings('default')


# In[23]:


pd.set_option('display.max_columns', 200)


# In[24]:


pd.set_option('display.max_rows', 100000)


# In[1]:


LOAD_DATA = False  # Set to True only in secure local environment

if LOAD_DATA:
    with open('./data/senior_ds_test/data/train/train_flag.csv', 'r') as f:
        df_flag = json.load(f)
else:
    df_flag = [] 


# In[26]:


df_flag.head()


# In[27]:


df_flag.columns


# In[28]:


df_flag['TARGET'].mean()


# ## Flag Data Split

# In[29]:


total_rows = len(df_flag)

# Calculate the size of each part
part_size = total_rows // 5
remainder = total_rows % 5

# Initialize a list to hold the parts
parts = []

# Split the dataset into 5 parts
start_idx = 0
for i in range(5):
    end_idx = start_idx + part_size + (1 if i < remainder else 0)  # Distribute remainder
    parts.append(df_flag.iloc[start_idx:end_idx])
    start_idx = end_idx

# Verify the split
flag1, flag2, flag3, flag4, flag5 = parts
for i, part in enumerate(parts):
    print(f"Flag {i+1} size: {part.shape}")


# In[30]:


flag1.shape


# In[31]:


flag1.nunique()


# In[32]:


df_flag.shape


# ## LOAD ACCOUNTS AND ENQUIRY DATA

# In[33]:


df_acc = pd.read_csv("/Users/aaniaadap/Desktop/HDFC Internship/df_acc.csv")


# In[35]:


df_acc.shape


# In[36]:


df_acc['uid'].nunique()


# In[37]:


df_enq = pd.read_csv("/Users/aaniaadap/Desktop/HDFC Internship/df_enq.csv")


# In[38]:


df_enq.shape


# ## MERGED DATA

# ### FLAG 1

# In[39]:


flag1['uid'].nunique()


# In[40]:


df1 = pd.merge(
    pd.merge(flag1, df_enq, on='uid', how='inner'),
    df_acc, on='uid', how='inner'
)


# In[41]:


df1.shape


# In[42]:


df1['uid'].nunique()


# In[ ]:


df1['uid'].value_counts()


# In[44]:


df1.head()


# In[ ]:


df1[df1['uid']=='XDA69787158']


# In[45]:


df1[df1.duplicated(subset='uid')].sort_values(by='uid')


# In[46]:


df1.shape


# ### FLAG 2

# In[47]:


df2 = pd.merge(
    pd.merge(flag2, df_enq, on='uid', how='inner'),
    df_acc, on='uid', how='inner'
)


# In[48]:


df2.shape


# In[49]:


df2['uid'].nunique()


# In[ ]:


df2['uid'].value_counts()


# In[51]:


df2.head()


# In[52]:


df2[df2.duplicated(subset='uid')].sort_values(by='uid')


# In[53]:


df2.shape


# ### FLAG 3

# In[54]:


df3 = pd.merge(
    pd.merge(flag3, df_enq, on='uid', how='inner'),
    df_acc, on='uid', how='inner'
)


# In[55]:


df3.shape


# In[56]:


df3['uid'].nunique()


# In[57]:


df3['uid'].value_counts()


# In[58]:


df3[df3.duplicated(subset='uid')].sort_values(by='uid')


# In[59]:


df3.shape


# ### FLAG 4

# In[60]:


df4 = pd.merge(
    pd.merge(flag4, df_enq, on='uid', how='inner'),
    df_acc, on='uid', how='inner'
)


# In[ ]:


df4.shape


# In[61]:


df4['uid'].nunique()


# In[62]:


df4.shape


# ### FLAG 5

# In[63]:


df5 = pd.merge(
    pd.merge(flag5, df_enq, on='uid', how='inner'),
    df_acc, on='uid', how='inner'
)


# In[64]:


df5.shape


# In[65]:


df5['uid'].nunique()


# In[66]:


df5[df5.duplicated(subset='uid')].sort_values(by='uid')


# In[67]:


df5.shape


# In[68]:


df5.head()


# ## MERGED DATASET

# In[69]:


df = pd.concat([df1,df2,df3,df4,df5])


# In[70]:


df.shape


# In[71]:


uids_flags = set(df_flag['uid'])
uids_df = set(df['uid'])


# In[72]:


missing_uids = uids_flags-uids_df


# In[73]:


len(missing_uids)


# In[74]:


df.columns


# In[75]:


user_dict = df[df["uid"]=='AAA09044550'].to_dict('records')


# In[76]:


user_dict


# In[77]:


df.to_csv("df_train.csv",index=False)


# ## Threshold

# In[79]:


threshold = 0.95

# Calculate percentage of NaNs
nan_percentage = df.isna().mean()

# Calculate percentage of zeros (excluding NaNs)
zero_percentage = (df == 0).sum() / df.shape[0]

# Calculate percentage of unique values
unique_percentage =  (df.nunique() / df.count())

# Identify columns with more than 95% NaNs, 0s, or unique values
columns_with_high_nan = nan_percentage[nan_percentage > threshold].index.tolist()
columns_with_high_zeros = zero_percentage[zero_percentage > threshold].index.tolist()
columns_with_high_unique = unique_percentage[unique_percentage > threshold].index.tolist()

print("Columns with more than 95% NaNs:", columns_with_high_nan)
print()
print("Columns with more than 95% zeros:", columns_with_high_zeros)
print()
print("Columns with more than 95% unique values:", columns_with_high_unique)


# In[80]:


threshold = 0.20

# Calculate percentage of NaNs
nan_percentage = df.isna().mean()

# Calculate percentage of zeros (excluding NaNs)
zero_percentage = (df == 0).sum() / df.shape[0]

# Calculate percentage of unique values
unique_percentage =  (df.nunique() / df.count())

# Identify columns with more than 95% NaNs, 0s, or unique values
columns_with_high_nan = nan_percentage[nan_percentage > threshold].index.tolist()
columns_with_high_zeros = zero_percentage[zero_percentage > threshold].index.tolist()
columns_with_less_than_20_percent_unique = unique_percentage[unique_percentage > threshold].index.tolist()

print("Columns with less than 20% NaNs:", columns_with_high_nan)
print()
print("Columns with less than 20% zeros:", columns_with_high_zeros)
print()
print("Columns with less than 20% unique values:", columns_with_less_than_20_percent_unique)


# In[81]:


len(columns_with_high_zeros)


# In[82]:


df.shape


# In[83]:


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


# In[84]:


df.shape


# In[85]:


columns_with_less_than_20_percent_unique


# In[86]:


"""num_rows = len(df)
for column in columns_with_less_than_20_percent_unique:
    if column in df.columns:
        df[column] = df[column].astype('category')
    else:
        print(f"Column '{column}' does not exist in the DataFrame.")
        """


# In[87]:


df.info()


# In[88]:


df.to_csv("df_train.csv",index=False)


# ## EDA

# ### Target

# In[89]:


target = df["TARGET"].value_counts().head(10)
plt.figure(figsize=(6,4))
plt.bar(target.index, target.values,width=0.3)
plt.xticks([0, 1], ['Good Loan', 'Bad Loan'],rotation=45)
plt.xlabel('Good/Bad Loan')
plt.ylabel('Count')
plt.title('Count of Good/Bad Loans')
plt.show()


# In[90]:


enq_type_counts = df["enqList_enquiry_type"].value_counts().head(10)
plt.figure(figsize=(10, 5))
plt.bar(enq_type_counts.index, enq_type_counts.values,width=0.3)
plt.xticks(rotation=45)
plt.xlabel('Enquiry Type')
plt.ylabel('Count')
plt.title('Count of Entries for Each Enquiry Type')
plt.show()


# In[91]:


plt.figure(figsize=(6, 4))
sns.barplot(x='NAME_CONTRACT_TYPE', y='enqList_enquiry_amt', data=df)
plt.xlabel('Contract Type')
plt.ylabel('Average Loan Amount')
plt.title('Average Loan Amount by Contract Type')
plt.show()


# In[ ]:


df_copy = df.copy()


# In[ ]:


bins = [0, 100000, 500000, 1000000, 5000000, 10000000]

labels = ['0-100k', '100k-500k', '500k-1M', '1M-5M','5M-10M']

df_copy['enqList_enquiry_amt_category'] = pd.cut(df_copy['enqList_enquiry_amt'], bins=bins, labels=labels)


# In[ ]:


#
category = df_copy["enqList_enquiry_amt_category"].value_counts()
sns.countplot(x='enqList_enquiry_amt_category', data=df_copy, palette='coolwarm')
plt.xlabel('Enquiry Amount Category')
plt.ylabel('Count')
plt.title('Distribution of Enquiry Data.ipynb Amounts by Category')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 6))
sns.histplot(df['enqList_enquiries_last_3_months'], bins=30, kde=True)
plt.xlabel('Number of Enquiries in the Last 3 Months')
plt.ylabel('Frequency')
plt.title('Distribution of Enquiries in the Last 3 Months')
plt.show()


# In[ ]:


Chris

