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

# In[1]:


import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_rows', 100000)


# In[3]:


pd.set_option('display.max_columns', 140)


# # Data Cleaning

# 
# ## Accounts Data Train

# In[4]:


LOAD_DATA = False  # Set to True only in secure local environment

if LOAD_DATA:
    with open('./data/senior_ds_test/data/train/accounts_data_train.json', 'r') as f:
        df = json.load(f)
else:
    df = []

flat_list = [item for sublist in df for item in sublist]
df_acc = pd.DataFrame(flat_list)


# In[5]:


df_acc.head()


# In[6]:


df_acc.columns


# In[7]:


df_acc.info()


# In[8]:


df_acc.isnull().sum()


# In[10]:


df_acc.nunique()


# In[11]:


df_acc["credit_type"].value_counts()


# In[12]:


df_acc["open_date"]=pd.to_datetime(df_acc["open_date"],format='%Y-%m-%d')
df_acc["closed_date"]=pd.to_datetime(df_acc["closed_date"],format='%Y-%m-%d')


# In[13]:


df_acc[df_acc.duplicated(subset=["uid",'open_date','closed_date',"loan_amount","credit_type"])].sort_values(by='uid')


# In[14]:


df_acc.duplicated(subset=["uid", "open_date", "closed_date","loan_amount","credit_type"]).sum()


# In[15]:


# Size of raw data
df_acc.shape


# In[16]:


#dropping duplicate rows
df_acc = df_acc.sort_values(by="uid").drop_duplicates(subset=["uid", "open_date", "closed_date","loan_amount"],keep='first')


# In[17]:


#Size after duplicated rows are dropped. - 7792 duplicates dropped. 
df_acc.shape


# In[18]:


# checking if loan amount column has null values.

df_acc.loc[df_acc["loan_amount"].isnull()]


# In[19]:


#dropping rows where loan amount has no value. - 3 rows

df_acc = df_acc.dropna(subset=['loan_amount'])


# In[20]:


df_acc.shape


# In[21]:


# checking if loan amount has values that are 0. 

df_acc[df_acc['loan_amount']==0].head(40)


# In[22]:


(df_acc['loan_amount']==0).sum()


# In[23]:


#dropping rows where loan amount is 0. 

df_acc = df_acc[df_acc['loan_amount']!=0]

#45773 rows removed.


# In[24]:


df_acc.shape


# In[25]:


#checking if closed date is earlier than the open date. 

df_acc[(df_acc['open_date'] > df_acc['closed_date'])]


# In[26]:


df_acc[(df_acc['open_date'] > df_acc['closed_date'])].shape


# In[27]:


df_acc = df_acc[~(df_acc['open_date'] > df_acc['closed_date'])]


df_acc.shape
# drops 12 rows. 


# In[28]:


#Feature 1 - Duration of account() in months. 

df_acc['duration_in_months'] = (df_acc['closed_date'] - df_acc['open_date']).dt.days / 30

df_acc.head()


# In[29]:


df_acc[(df_acc['duration_in_months'] > 1) & (df_acc['payment_hist_string'].isnull())]


# In[30]:


df_acc[(df_acc['duration_in_months'] > 1) & (df_acc['payment_hist_string']=='')]


# In[31]:


df_acc[df_acc['open_date'].isnull()]


# In[32]:


#no. of open accounts. 

print("Number of open accounts:")
df_acc['closed_date'].isnull().sum()


# In[33]:


df_acc.shape


# In[34]:


df_acc.columns


# In[35]:


df_acc.loc[0]


# ### Accounts Features

# ##### DURATION 

# In[36]:


duration_aggregates = df_acc.groupby('uid')['duration_in_months'].agg(['mean', 'sum', 'min', 'max']).reset_index()
duration_aggregates.columns = ['uid', 'mean_duration_months', 'total_duration_months', 'min_duration_months', 'max_duration_months']

# Merge the aggregates back into df_acc
df_acc = df_acc.merge(duration_aggregates, on='uid', how='left')


# In[37]:


df_acc.shape


# ##### TIME BASED

# In[38]:


print(df_acc["loan_amount"].describe())
print()
print('Maximum:',df_acc["loan_amount"].max())
print('Minimum:',df_acc["loan_amount"].min())


# In[39]:


earliest_acc = df_acc.groupby('uid')['open_date'].min().reset_index()
earliest_acc.columns = ['uid', 'earliest_acc_date']

# Extract year, month, and day from earliest_acc_date
earliest_acc['earliest_acc_year'] = earliest_acc['earliest_acc_date'].dt.year
earliest_acc['earliest_acc_month'] = earliest_acc['earliest_acc_date'].dt.month
earliest_acc['earliest_acc_day'] = earliest_acc['earliest_acc_date'].dt.day

# Merge the extracted columns back into df_acc
df_acc = df_acc.merge(earliest_acc[['uid', 'earliest_acc_year', 'earliest_acc_month', 'earliest_acc_day']], on='uid', how='left')


# In[40]:


latest_acc = df_acc.groupby('uid')['open_date'].max().reset_index()
latest_acc.columns = ['uid', 'latest_acc_date']

# Extract year, month, and day from latest_acc_date
latest_acc['latest_acc_year'] = latest_acc['latest_acc_date'].dt.year
latest_acc['latest_acc_month'] = latest_acc['latest_acc_date'].dt.month
latest_acc['latest_acc_day'] = latest_acc['latest_acc_date'].dt.day

df_acc = df_acc.merge(latest_acc[['uid', 'latest_acc_year', 'latest_acc_month', 'latest_acc_day']], on='uid', how='left')


# In[41]:


df_acc.head()


# In[42]:


df_acc.shape


# ##### LOAN AMOUNT

# In[43]:


#loan_amount_categories.
bins = [0, 100000, 500000, 1000000, 5000000, 10000000, 50000000,100000000,500000000]

labels = ['0-100k', '100k-500k', '500k-1M', '1M-5M', '5M-10M', '10M-50M','50M-100M','100M-500M']

df_acc['loan_amount_category'] = pd.cut(df_acc['loan_amount'], bins=bins, labels=labels)
df_acc.head()


# In[ ]:


#Total of each category.
df_acc["loan_amount_category"].value_counts()


# In[44]:


# Metrics on Loan Amount
df_acc["avg_loan_amount"] = df_acc["uid"].map(df_acc.groupby("uid")["loan_amount"].mean())
df_acc["median_loan_amount"] = df_acc["uid"].map(df_acc.groupby("uid")["loan_amount"].median())
df_acc["max_loan_amt"] = df_acc["uid"].map(df_acc.groupby("uid")["loan_amount"].max())
df_acc["min_loan_amt"] = df_acc["uid"].map(df_acc.groupby("uid")["loan_amount"].min())


# In[45]:


#avg. payment to be made each month. 
df_acc["avg_monthly_payment"] = (df_acc["loan_amount"]/df_acc["duration_in_months"]).round(2)


# In[46]:


df_acc.shape


# In[47]:


df_acc.columns


# #### OVERDUES

# In[48]:


# Metrics on Amount Overdue
df_acc["avg_overdues"] = df_acc["uid"].map(df_acc.groupby("uid")["amount_overdue"].mean())
df_acc["median_overdues"] = df_acc["uid"].map(df_acc.groupby("uid")["amount_overdue"].median())
df_acc["max_overdues"] = df_acc["uid"].map(df_acc.groupby("uid")["amount_overdue"].max())
df_acc["min_overdues"] = df_acc["uid"].map(df_acc.groupby("uid")["amount_overdue"].min())


# In[49]:


df_acc.shape


# #### PAYMENT HISTORY

# In[50]:


#Payment details
def analyze_payment_history(payment_history):
    months = [payment_history[i:i+3] for i in range(0, len(payment_history), 3)]

    late_payment_count = sum(1 for month in months if month != "000")
    max_consecutive_late_payments = 0
    consecutive_late_payments = 0
    last_payment_was_late = False
    
    for month in months:
        if month != "000":
            consecutive_late_payments = consecutive_late_payments + 1 if last_payment_was_late else 1
            last_payment_was_late = True
        else:
            max_consecutive_late_payments = max(max_consecutive_late_payments, consecutive_late_payments)
            consecutive_late_payments = 0
            last_payment_was_late = False
    
    max_consecutive_late_payments = max(max_consecutive_late_payments, consecutive_late_payments)
    on_time_payment_count = len(months) - late_payment_count
    time_since_last_late_payment = next((i for i, month in enumerate(reversed(months)) if month != "000"), len(months))

    return {
        'late_payments': late_payment_count,
        'on_time_payments': on_time_payment_count,
        'max_consecutive_late_payments': max_consecutive_late_payments,
        'time_since_last_late_payment': time_since_last_late_payment
    }
# Apply the function to the 'payment_hist_string' column and expand the result into separate columns
df_analysis = df_acc['payment_hist_string'].apply(analyze_payment_history).apply(pd.Series)

# Concatenate the results back to the original DataFrame
df_acc = pd.concat([df_acc, df_analysis], axis=1)


# In[51]:


# metrics for on time payments 
df_acc["avg_ontime_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['on_time_payments'].mean().round(2))
df_acc["median_ontime_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['on_time_payments'].median())
df_acc["max_ontime_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['on_time_payments'].max())
df_acc["min_ontime_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['on_time_payments'].min())


# In[52]:


# metrics for late payments 
df_acc["avg_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['late_payments'].mean().round(2))
df_acc["median_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['late_payments'].median())
df_acc["max_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['late_payments'].max())
df_acc["min_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['late_payments'].min())


# In[53]:


# metrics for max_consecutive_late_payments
df_acc["avg_consecutive_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['max_consecutive_late_payments'].mean().round(2))
df_acc["median_consecutive_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['max_consecutive_late_payments'].median())
df_acc["max_consecutive_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['max_consecutive_late_payments'].max())
df_acc["min_consecutive_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['max_consecutive_late_payments'].min())


# In[54]:


df_acc["max_consecutive_late_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['max_consecutive_late_payments'].max())


# In[55]:


df_acc.head()


# In[56]:


df_acc.shape


# #### DELINQUENCY

# In[57]:


def calculate_delinquency_metrics(payment_history_string, delinquency_threshold=60):
    try:
        total_payments = len(payment_history_string) // 3
        if total_payments == 0:
            return None, None, None, None  # No payments to evaluate
        
        delinquent_payments = 0
        total_days_past_due = 0

        for i in range(0, len(payment_history_string), 3):
            payment_status = payment_history_string[i:i+3]
            days_past_due = int(payment_status)

            total_days_past_due += days_past_due

            if days_past_due > delinquency_threshold:
                delinquent_payments += 1

        delinquency_rate = delinquent_payments / total_payments
    except ZeroDivisionError:
        delinquency_rate = None
    except ValueError:  # if there's an invalid payment history format
        delinquency_rate = None
        total_payments = None
        delinquent_payments = None
        total_days_past_due = None

    delinquency_rate = delinquency_rate * 100 if delinquency_rate is not None else None
    
    return delinquency_rate, total_payments, delinquent_payments, total_days_past_due

# Apply the function to the 'payment_hist_string' column
df_acc[['delinquency_rate', 'total_payments', 'delinquent_payments', 'total_DPD']] = df_acc['payment_hist_string'].apply(
    lambda x: pd.Series(calculate_delinquency_metrics(x))
)



# In[58]:


# metrics for Total Payments
df_acc["avg_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['total_payments'].mean().round(2))
df_acc["median_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['total_payments'].median())
df_acc["max_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['total_payments'].max())
df_acc["min_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['total_payments'].min())


# In[59]:


# metrics for Delinquent Payments
df_acc["avg_del_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquent_payments'].mean().round(2))
df_acc["median_del_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquent_payments'].median())
df_acc["max_del_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquent_payments'].max())
df_acc["min_del_payments"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquent_payments'].min())


# In[60]:


# metrics for Days Past Due
df_acc["avg_DPD"] = df_acc["uid"].map(df_acc.groupby('uid')['total_DPD'].mean().round(2))
df_acc["median_DPD"] = df_acc["uid"].map(df_acc.groupby('uid')['total_DPD'].median())
df_acc["max_DPD"] = df_acc["uid"].map(df_acc.groupby('uid')['total_DPD'].max())
df_acc["min_DPD"] = df_acc["uid"].map(df_acc.groupby('uid')['total_DPD'].min())


# In[61]:


# metrics for Deliquency Rates
df_acc["avg_DR"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquency_rate'].mean().round(2))
df_acc["median_DR"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquency_rate'].median())
df_acc["max_DR"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquency_rate'].max())
df_acc["min_DR"] = df_acc["uid"].map(df_acc.groupby('uid')['delinquency_rate'].min())


# In[62]:


df_acc.shape


# In[63]:


df_acc.head()


# #### MONTHLY SEGGREGATION OF PAYMENT HISTORY

# In[64]:


# splitting payment history string acc to last 3 mons, 6 mons, and 9 mons. 
def split_payment_history(payment_hist_string):
    last_1_month = None
    last_3_months = None
    last_6_months = None
    last_9_months = None
    if len(payment_hist_string) >= 3:
        last_1_month = payment_hist_string[:3]
    if len(payment_hist_string) >= 3*3:
        last_3_months = payment_hist_string[:3*3]
    if len(payment_hist_string) >= 6*3:
        last_6_months = payment_hist_string[:6*3]
    if len(payment_hist_string) >= 9*3:
        last_9_months = payment_hist_string[:9*3]
    
    return pd.Series([last_1_month,last_3_months, last_6_months, last_9_months])

# Apply the function to the 'payment_hist_string' column
df_acc[["last_1_month",'last_3_months', 'last_6_months', 'last_9_months']] = df_acc['payment_hist_string'].apply(split_payment_history)


# ##### PAST 1 MONTH

# In[65]:


#avg delinquency in latest month.
def past_month_delinquency(last_1_month):
    if last_1_month and last_1_month.isdigit():
        return int(last_1_month)
    return 0

df_acc['DPD_last_1_month'] = df_acc['last_1_month'].apply(past_month_delinquency)


# In[67]:


#deliquency over the past month
df_acc["avg_DPD_last_1_month"] = df_acc["uid"].map(df_acc.groupby('uid')['DPD_last_1_month'].mean().round(2))
df_acc["median_DPD_last_1_month"] = df_acc["uid"].map(df_acc.groupby('uid')['DPD_last_1_month'].median())
df_acc["max_DPD_last_1_month"] = df_acc["uid"].map(df_acc.groupby('uid')['DPD_last_1_month'].max())
df_acc["min_DPD_last_1_month"] = df_acc["uid"].map(df_acc.groupby('uid')['DPD_last_1_month'].min())


# In[68]:


df_acc.shape


# ##### PAST 3,6,9 MONTHS

# In[69]:


#delinquency for last,3,6 and 9 months.

def delinquency(last_x_month):
    if pd.isna(last_x_month):
        return 0
    total_days = 0
    for i in range(0, len(last_x_month), 3):
        total_days += int(last_x_month[i:i+3])
    return total_days


df_acc["total_delinquency_3_mons"] = df_acc["last_3_months"].apply(delinquency)
df_acc["avg_DPD_3_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_3_mons'].mean().round(2))
df_acc["median_DPD_3_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_3_mons'].median())
df_acc["max_DPD_3_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_3_mons'].max())
df_acc["min_DPD_3_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_3_mons'].min())


df_acc["total_delinquency_6_mons"] = df_acc["last_6_months"].apply(delinquency)
df_acc["avg_DPD_6_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_6_mons'].mean().round(2))
df_acc["median_DPD_6_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_6_mons'].median())
df_acc["max_DPD_6_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_6_mons'].max())
df_acc["min_DPD_6_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_6_mons'].min())

df_acc["total_delinquency_9_mons"] = df_acc["last_9_months"].apply(delinquency)
df_acc["avg_DPD_9_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_9_mons'].mean().round(2))
df_acc["max_DPD_9_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_9_mons'].max())
df_acc["median_DPD_9_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_9_mons'].median())
df_acc["min_DPD_9_mons"] = df_acc["uid"].map(df_acc.groupby('uid')['total_delinquency_9_mons'].min())


# In[ ]:


print(df_acc["delinquency_rate"].describe())


print()
print("Max. Delinquency: ",df_acc["delinquency_rate"].max())
print("Min. Delinquency: ",df_acc["delinquency_rate"].min())


# In[ ]:


df_acc[df_acc["amount_overdue"]!=0].head(40)


# In[ ]:


df_non_zero_dpd = df_acc.query("DPD_last_1_month != 0")


# In[70]:


df_acc.shape


# #### CREDIT TYPES

# In[71]:


pivot_mean = pd.pivot_table(df_acc, 
                            index='uid', 
                            columns='credit_type', 
                            values='loan_amount', 
                            aggfunc='mean', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_mean.columns = [f'{credit}_mean_loan' for credit in pivot_mean.columns]

# Reset the index to merge with the original dataframe
pivot_mean.reset_index(inplace=True)

df_acc = df_acc.merge(pivot_mean, on='uid', how='left')


# In[72]:


df_acc.shape


# In[73]:


pivot_median = pd.pivot_table(df_acc, 
                            index='uid', 
                            columns='credit_type', 
                            values='loan_amount', 
                            aggfunc='median', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_median.columns = [f'{credit}_median_loan' for credit in pivot_median.columns]

# Reset the index to merge with the original dataframe
pivot_median.reset_index(inplace=True)

df_acc = df_acc.merge(pivot_median, on='uid', how='left')


# In[74]:


df_acc.shape


# In[75]:


pivot_sum = pd.pivot_table(df_acc, 
                            index='uid', 
                            columns='credit_type', 
                            values='loan_amount', 
                            aggfunc='sum', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_sum.columns = [f'{credit}_total_loan' for credit in pivot_sum.columns]

# Reset the index to merge with the original dataframe
pivot_sum.reset_index(inplace=True)

df_acc = df_acc.merge(pivot_sum, on='uid', how='left')


# In[197]:


df_acc.head()


# In[76]:


df_acc.shape


# #### LOAN STATUS

# In[77]:


#Loan Status
df_acc['loan_status'] = df_acc['closed_date'].apply(lambda x: 'Open' if pd.isnull(x) else 'Closed')


# In[78]:


df_acc.head()


# In[79]:


df_acc.shape


# In[80]:


most_frequent_credit_type = df_acc.groupby('uid')['credit_type'].agg(lambda x: x.value_counts().idxmax())

df_acc['most_frequent_credit_type'] = df_acc['uid'].map(most_frequent_credit_type)


# In[ ]:


df_acc.columns


# ### Feature Selection

# In[81]:


columns_to_drop = [
    "credit_type",
    "loan_amount",
    "amount_overdue",
    "open_date",
    "closed_date",
    "payment_hist_string",
    "duration_in_months",
    "late_payments",
    "on_time_payments",
    "time_since_last_late_payment",
    "max_consecutive_late_payments",
    "delinquency_rate",
    "total_payments",
    "delinquent_payments",
    "total_DPD",
    "loan_amount_category",
    "median_ontime_payments",
    "median_late_payments",
    "median_consecutive_late_payments",
    "median_DR","max_DR","min_DR",
    "last_1_month",
    "last_3_months",
    "last_6_months",
    "last_9_months",
    "avg_monthly_payment",
    "total_delinquency_3_mons",
    "total_delinquency_6_mons",
    "total_delinquency_9_mons",
    "DPD_last_1_month",
    "median_DPD_last_1_month",
    "max_DPD_last_1_month",
    "min_DPD_last_1_month",
    "median_DPD_3_mons",
    "max_DPD_3_mons",
    "min_DPD_3_mons",
    "median_DPD_6_mons",
    "max_DPD_6_mons",
    "min_DPD_6_mons",
    "median_DPD_9_mons",
    "max_DPD_9_mons",
    "min_DPD_9_mons",
    "median_overdues"
    
]


df_acc = df_acc.drop(columns=columns_to_drop)


# In[82]:


df_acc.shape


# In[83]:


df_acc["uid"].nunique()


# In[84]:


df_acc.duplicated(subset="uid")


# In[85]:


df_acc = df_acc.drop_duplicates(subset="uid",keep="first")


# In[86]:


df_acc.head()


# In[87]:


acc_prefix = 'accList_'
df_acc = df_acc.rename(columns=lambda x: acc_prefix + x if x != 'uid' else x)


# In[88]:


user_dict = df_acc[df_acc["uid"]=='AAA09044550'].to_dict('records')


# In[89]:


user_dict


# In[90]:


df_acc.to_csv('df_acc.csv', index=False)


# # EDA
# 

# In[106]:


df_acc.columns


# ## Credit Type

# In[ ]:


credit_type_counts = df_acc["credit_type"].value_counts().head(5)
plt.figure(figsize=(6, 5))
plt.bar(credit_type_counts.index, credit_type_counts.values,width=0.3)
plt.xticks(rotation=45)
plt.xlabel('Credit Type')
plt.ylabel('Count')
plt.title('Count of Entries for Each Credit Type')
plt.show()


# Most popular type of loan/credit is 'Consumer Credit'

# In[ ]:


filt = df_acc[df_acc["amount_overdue"]!=0]

# Box plot for amount overdue by credit type
plt.figure(figsize=(12, 6))
sns.boxplot(x='credit_type', y='amount_overdue', data=filt, palette='pastel')

plt.xlabel('Credit Type')
plt.xticks(rotation=45)
plt.ylabel('Amount Overdue')
plt.title('Amount Overdue by Credit Type')

plt.show()


# In[ ]:


df_acc.sort_values(by="amount_overdue",ascending=False).head(20)


# **Most of the overdue amount for consumer credit and credit card is concentrated between the range of 0 to 50000000.**

# In[ ]:


# Delinquency Rate
filt=df_acc[df_acc["delinquency_rate"]!=0]

plt.figure(figsize=(15, 5))
sns.scatterplot(x='credit_type', y='delinquency_rate', data=filt,alpha=0.3)
plt.title('Delinquency Rate by Credit Type')
plt.xlabel('Credit Type')
plt.ylabel('Delinquency Rate')
plt.xticks(rotation=45)
plt.show()


# ## Delinquency Rate

# In[ ]:


avg_delinquency = df_acc.groupby("credit_type")["delinquency_rate"].mean().reset_index().sort_values('delinquency_rate',ascending=False).head(10)


# In[ ]:


plt.figure(figsize=(20,5))
plt.bar(avg_delinquency["credit_type"], avg_delinquency["delinquency_rate"], color='pink',width=0.4)
plt.xticks(rotation=45)
plt.xlabel("Credit Type")
plt.ylabel("Average Delinquency")
plt.title("Delinquency Rate across Credit Types")
plt.grid(True)
plt.show()


# Maximum Deliquency is shown by customers who have taken a cash loan.

# In[ ]:


df_acc['open_year'] = df_acc["open_date"].dt.year

avg_delinquency = df_acc.groupby("open_year")["delinquency_rate"].mean()
plt.plot(avg_delinquency.index, avg_delinquency.values, marker='o', linestyle='-', color='green')
plt.xlabel('Open Date')
plt.ylabel('Delinquency Rate')
plt.title('Delinquency Rate Over Time')
plt.show()


# **Highest delinquency is potraye by customers who have opened accounts in 2013.**
# 
# **There is a sudden drop in delinquency rate between 2019 to 2020.**
# 
# **Also Delinquecy decrease as the opening date near the present time.**

# In[ ]:


monthly_delinquency = df_acc.groupby([df_acc['open_year'], df_acc['month_name']])['delinquency_rate'].mean().reset_index()

print(monthly_delinquency.max())


# In[ ]:


plt.figure(figsize=(12, 6))
plt.bar(monthly_delinquency['month_name'], monthly_delinquency['delinquency_rate'])
plt.xlabel('Month')
plt.ylabel('Delinquency Rate (%)')
plt.title('Monthly Delinquency Rate over Time')
plt.xticks(rotation=45)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.grid(True)
plt.show()


# **Highest delinquency are potrayed by customers who open accounts in the month of October.**

# In[ ]:


avg_late_payments = df_acc.groupby('late_payments')['delinquency_rate'].mean()

plt.figure(figsize=(10, 6))
plt.plot(avg_late_payments.index, avg_late_payments.values) 
plt.xlabel("Average Number of Late Payments")
plt.ylabel("Delinquency Rate")
plt.title("Delinquency Rate vs. Average Late Payments")
plt.grid(True)
plt.show()


# ## Date

# In[ ]:


a = df_acc.groupby(df_acc['open_date'].dt.month_name())['uid'].nunique().sort_values()
b = df_acc.groupby(df_acc['closed_date'].dt.month_name())['uid'].nunique().sort_values()


plt.subplot(1,2,1)
a.plot(kind='bar', label='Opened Accounts',color='r',alpha=0.3)
plt.xlabel('Month')
plt.ylabel('Number of Accounts')
plt.title('Opened Loan Accounts by Month')

plt.subplot(1,2,2)
b.plot(kind='bar', label='Closed Accounts',color='b',alpha=0.3)
plt.xlabel('Month')
plt.ylabel('Number of Accounts')
plt.title('Closed Loan Accounts by Month')

plt.tight_layout()
plt.show()


# In[ ]:


plt.bar(['On-Time Payments', 'Late Payments'], [df_acc['on_time_payments'].sum(), df_acc['late_payments'].sum()])
plt.xlabel('Payment Type')
plt.ylabel('Total Count')
plt.title('On-Time vs. Late Payments (Total Count)')
plt.show()


# **Late payments are significantly lower than on time payments.**

# In[ ]:


plt.figure(figsize=(10, 6))
sns.histplot(df_acc['time_since_last_late_payment'], kde=True)
plt.title('Distribution of Time Since Last Late Payment')
plt.xlabel('Time Since Last Late Payment (days)')
plt.ylabel('Frequency')
plt.show()


# **Right Skewed distribution suggests that,, majority of late payments were made recently i.e. less than 20 days.**

# ## Day

# In[ ]:


# Box plot for loan amount by day opened
plt.figure(figsize=(14, 6))
sns.boxplot(x='day_opened', y='loan_amount', data=df_acc, palette='coolwarm')

# Adding labels and title
plt.xlabel('Day Opened')
plt.ylabel('Loan Amount')
plt.title('Loan Amount Distribution by Day Opened')

# Display the plot
plt.show()


# **Majority of loans taken lie between 0 - 100000000.. very few payments go past 1CR. They have no uniqueness wrt to the days of the week**

# ## Loan Amount Category

# In[ ]:


#understanding distribution of different categories. 
category = df_acc["loan_amount_category"].value_counts()
sns.countplot(x='loan_amount_category', data=df_acc, palette='coolwarm')
plt.xlabel('Loan Amount Category')
plt.ylabel('Count')
plt.title('Distribution of Loan Amounts by Category')
plt.xticks(rotation=45)
plt.show()


# Maximum loans range btw 100k to 500k. 
