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

# In[3]:


import json
import pandas as pd


# In[ ]:


pd.set_option('display.max_rows', 100000)


# In[ ]:


pd.set_option('display.max_columns', 140)


# In[9]:


LOAD_DATA = False  # Set to True only in secure local environment

if LOAD_DATA:
    with open('./data/senior_ds_test/data/test/accounts_data_test.json', 'r') as f:
        df = json.load(f)
else:
    df = []


flat_list = [item for sublist in df for item in sublist]
df_acc_test = pd.DataFrame(flat_list)


# In[10]:


df_acc_test.shape


# In[158]:


df_acc_test.info()


# In[159]:


df_acc_test.describe()


# In[160]:


df_acc_test.nunique()


# In[161]:


df_acc_test.columns


# In[162]:


df_acc_test["credit_type"].value_counts()


# In[163]:


df_acc_test["open_date"]=pd.to_datetime(df_acc_test["open_date"],format='%Y-%m-%d')
df_acc_test["closed_date"]=pd.to_datetime(df_acc_test["closed_date"],format='%Y-%m-%d')


# In[164]:


df_acc_test[df_acc_test.duplicated(subset=["uid",'open_date','closed_date',"loan_amount","credit_type"])].sort_values(by='uid').head(100)


# In[165]:


df_acc_test.duplicated(subset=["uid", "open_date", "closed_date","loan_amount","credit_type"]).sum()


# In[166]:


df_acc_test = df_acc_test.sort_values(by="uid").drop_duplicates(subset=["uid", "open_date", "closed_date","loan_amount"],keep='first')


# In[167]:


df_acc_test.shape


# In[168]:


df_acc_test.loc[df_acc_test["loan_amount"].isnull()]


# In[169]:


df_acc_test[df_acc_test['loan_amount']==0].head()


# In[170]:


(df_acc_test['loan_amount']==0).sum()


# In[171]:


df_acc_test = df_acc_test[df_acc_test['loan_amount']!=0]


# In[172]:


df_acc_test.shape


# In[173]:


#checking if closed date is earlier than the open date. 

df_acc_test[(df_acc_test['open_date'] > df_acc_test['closed_date'])]


# In[174]:


df_acc_test[(df_acc_test['open_date'] > df_acc_test['closed_date'])].shape


# In[175]:


df_acc_test = df_acc_test[~(df_acc_test['open_date'] > df_acc_test['closed_date'])]


df_acc_test.shape
# drops 1 rows. 


# In[176]:


#Feature 1 - Duration of account() in months. 

df_acc_test['duration_in_months'] = (df_acc_test['closed_date'] - df_acc_test['open_date']).dt.days / 30

df_acc_test.head()


# In[177]:


df_acc_test[(df_acc_test['duration_in_months'] > 1) & (df_acc_test['payment_hist_string'].isnull())]


# In[178]:


df_acc_test[(df_acc_test['duration_in_months'] > 1) & (df_acc_test['payment_hist_string']=='')]


# In[179]:


df_acc_test[df_acc_test['open_date'].isnull()]


# In[180]:


print("Number of open accounts:")
df_acc_test['closed_date'].isnull().sum()


# In[181]:


df_acc_test.shape


# ## Accounts Features

# ### Duration Aggregates

# In[182]:


duration_aggregates = df_acc_test.groupby('uid')['duration_in_months'].agg(['mean', 'sum', 'min', 'max']).reset_index()
duration_aggregates.columns = ['uid', 'mean_duration_months', 'total_duration_months', 'min_duration_months', 'max_duration_months']

# Merge the aggregates back into df_acc
df_acc_test = df_acc_test.merge(duration_aggregates, on='uid', how='left')


# In[183]:


df_acc_test.shape


# ### TIME BASED

# In[184]:


print(df_acc_test["loan_amount"].describe())
print()
print('Maximum:',df_acc_test["loan_amount"].max())
print('Minimum:',df_acc_test["loan_amount"].min())


# In[185]:


earliest_acc = df_acc_test.groupby('uid')['open_date'].min().reset_index()
earliest_acc.columns = ['uid', 'earliest_acc_date']

# Extract year, month, and day from earliest_acc_date
earliest_acc['earliest_acc_year'] = earliest_acc['earliest_acc_date'].dt.year
earliest_acc['earliest_acc_month'] = earliest_acc['earliest_acc_date'].dt.month
earliest_acc['earliest_acc_day'] = earliest_acc['earliest_acc_date'].dt.day

# Merge the extracted columns back into df_acc
df_acc_test = df_acc_test.merge(earliest_acc[['uid', 'earliest_acc_year', 'earliest_acc_month', 'earliest_acc_day']], on='uid', how='left')


# In[186]:


latest_acc = df_acc_test.groupby('uid')['open_date'].max().reset_index()
latest_acc.columns = ['uid', 'latest_acc_date']

# Extract year, month, and day from latest_acc_date
latest_acc['latest_acc_year'] = latest_acc['latest_acc_date'].dt.year
latest_acc['latest_acc_month'] = latest_acc['latest_acc_date'].dt.month
latest_acc['latest_acc_day'] = latest_acc['latest_acc_date'].dt.day

df_acc_test = df_acc_test.merge(latest_acc[['uid', 'latest_acc_year', 'latest_acc_month', 'latest_acc_day']], on='uid', how='left')


# In[187]:


df_acc_test.shape


# ### Loan Amount Categories

# In[188]:


#loan_amount_categories.
bins = [0, 100000, 500000, 1000000, 5000000, 10000000, 50000000,100000000,500000000]

labels = ['0-100k', '100k-500k', '500k-1M', '1M-5M', '5M-10M', '10M-50M','50M-100M','100M-500M']

df_acc_test['loan_amount_category'] = pd.cut(df_acc_test['loan_amount'], bins=bins, labels=labels)
df_acc_test.head()


# In[189]:


df_acc_test["loan_amount_category"].value_counts()


# In[190]:


# Metrics on Loan Amount
df_acc_test["avg_loan_amount"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["loan_amount"].mean())
df_acc_test["median_loan_amount"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["loan_amount"].median())
df_acc_test["max_loan_amt"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["loan_amount"].max())
df_acc_test["min_loan_amt"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["loan_amount"].min())


# In[191]:


df_acc_test["avg_monthly_payment"] = (df_acc_test["loan_amount"]/df_acc_test["duration_in_months"]).round(2)


# In[192]:


df_acc_test.shape


# ### Overdues
# 

# In[193]:


df_acc_test["avg_overdues"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["amount_overdue"].mean())
df_acc_test["median_overdues"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["amount_overdue"].median())
df_acc_test["max_overdues"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["amount_overdue"].max())
df_acc_test["min_overdues"] = df_acc_test["uid"].map(df_acc_test.groupby("uid")["amount_overdue"].min())


# ### PAYMENT HISTORY

# In[194]:


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
df_analysis = df_acc_test['payment_hist_string'].apply(analyze_payment_history).apply(pd.Series)

# Concatenate the results back to the original DataFrame
df_acc_test = pd.concat([df_acc_test, df_analysis], axis=1)


# In[195]:


# metrics for on time payments 
df_acc_test["avg_ontime_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['on_time_payments'].mean().round(2))
df_acc_test["median_ontime_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['on_time_payments'].median())
df_acc_test["max_ontime_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['on_time_payments'].max())
df_acc_test["min_ontime_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['on_time_payments'].min())


# In[196]:


# metrics for late payments 
df_acc_test["avg_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['late_payments'].mean().round(2))
df_acc_test["median_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['late_payments'].median())
df_acc_test["max_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['late_payments'].max())
df_acc_test["min_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['late_payments'].min())


# In[197]:


# metrics for max_consecutive_late_payments
df_acc_test["avg_consecutive_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['max_consecutive_late_payments'].mean().round(2))
df_acc_test["median_consecutive_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['max_consecutive_late_payments'].median())
df_acc_test["max_consecutive_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['max_consecutive_late_payments'].max())
df_acc_test["min_consecutive_late_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['max_consecutive_late_payments'].min())


# In[198]:


df_acc_test.shape


# In[199]:


df_acc_test.head()


# ### DElINQUENCY RATE

# In[200]:


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
df_acc_test[['delinquency_rate', 'total_payments', 'delinquent_payments', 'total_DPD']] = df_acc_test['payment_hist_string'].apply(
    lambda x: pd.Series(calculate_delinquency_metrics(x))
)



# In[201]:


# metrics for Total Payments
df_acc_test["avg_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_payments'].mean().round(2))
df_acc_test["median_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_payments'].median())
df_acc_test["max_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_payments'].max())
df_acc_test["min_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_payments'].min())


# In[202]:


df_acc_test["avg_del_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquent_payments'].mean().round(2))
df_acc_test["median_del_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquent_payments'].median())
df_acc_test["max_del_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquent_payments'].max())
df_acc_test["min_del_payments"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquent_payments'].min())


# In[203]:


# metrics for Days Past Due
df_acc_test["avg_DPD"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_DPD'].mean().round(2))
df_acc_test["median_DPD"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_DPD'].median())
df_acc_test["max_DPD"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_DPD'].max())
df_acc_test["min_DPD"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_DPD'].min())


# In[204]:


# metrics for Deliquency Rates
df_acc_test["avg_DR"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquency_rate'].mean().round(2))
df_acc_test["median_DR"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquency_rate'].median())
df_acc_test["max_DR"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquency_rate'].max())
df_acc_test["min_DR"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['delinquency_rate'].min())


# In[205]:


df_acc_test.shape


# ### MONTHLY SEGGREGATION OF PAYMENT HISTORY

# In[206]:


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
df_acc_test[["last_1_month",'last_3_months', 'last_6_months', 'last_9_months']] = df_acc_test['payment_hist_string'].apply(split_payment_history)


# #### PAST 1 MONTH

# In[207]:


#avg delinquency in latest month.
def past_month_delinquency(last_1_month):
    if last_1_month and last_1_month.isdigit():
        return int(last_1_month)
    return 0

df_acc_test['DPD_last_1_month'] = df_acc_test['last_1_month'].apply(past_month_delinquency)


# In[208]:


#deliquency over the past month
df_acc_test["avg_DPD_last_1_month"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['DPD_last_1_month'].mean().round(2))
df_acc_test["median_DPD_last_1_month"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['DPD_last_1_month'].median())
df_acc_test["max_DPD_last_1_month"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['DPD_last_1_month'].max())
df_acc_test["min_DPD_last_1_month"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['DPD_last_1_month'].min())


# In[209]:


df_acc_test.shape


# #### PAST 3,6,9 MONTHS

# In[210]:


#delinquency for last,3,6 and 9 months.

def delinquency(last_x_month):
    if pd.isna(last_x_month):
        return 0
    total_days = 0
    for i in range(0, len(last_x_month), 3):
        total_days += int(last_x_month[i:i+3])
    return total_days


df_acc_test["total_delinquency_3_mons"] = df_acc_test["last_3_months"].apply(delinquency)
df_acc_test["avg_DPD_3_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_3_mons'].mean().round(2))
df_acc_test["median_DPD_3_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_3_mons'].median())
df_acc_test["max_DPD_3_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_3_mons'].max())
df_acc_test["min_DPD_3_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_3_mons'].min())


df_acc_test["total_delinquency_6_mons"] = df_acc_test["last_6_months"].apply(delinquency)
df_acc_test["avg_DPD_6_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_6_mons'].mean().round(2))
df_acc_test["median_DPD_6_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_6_mons'].median())
df_acc_test["max_DPD_6_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_6_mons'].max())
df_acc_test["min_DPD_6_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_6_mons'].min())

df_acc_test["total_delinquency_9_mons"] = df_acc_test["last_9_months"].apply(delinquency)
df_acc_test["avg_DPD_9_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_9_mons'].mean().round(2))
df_acc_test["max_DPD_9_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_9_mons'].max())
df_acc_test["median_DPD_9_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_9_mons'].median())
df_acc_test["min_DPD_9_mons"] = df_acc_test["uid"].map(df_acc_test.groupby('uid')['total_delinquency_9_mons'].min())


# In[211]:


print(df_acc_test["delinquency_rate"].describe())


print()
print("Max. Delinquency: ",df_acc_test["delinquency_rate"].max())
print("Min. Delinquency: ",df_acc_test["delinquency_rate"].min())


# ### CREDIT TYPES

# In[212]:


pivot_mean = pd.pivot_table(df_acc_test, 
                            index='uid', 
                            columns='credit_type', 
                            values='loan_amount', 
                            aggfunc='mean', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_mean.columns = [f'{credit}_mean_loan' for credit in pivot_mean.columns]

# Reset the index to merge with the original dataframe
pivot_mean.reset_index(inplace=True)

df_acc_test = df_acc_test.merge(pivot_mean, on='uid', how='left')


# In[213]:


df_acc_test.shape


# In[214]:


df_acc_test.head()


# In[215]:


pivot_median = pd.pivot_table(df_acc_test, 
                            index='uid', 
                            columns='credit_type', 
                            values='loan_amount', 
                            aggfunc='median', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_median.columns = [f'{credit}_median_loan' for credit in pivot_median.columns]

# Reset the index to merge with the original dataframe
pivot_median.reset_index(inplace=True)

df_acc_test = df_acc_test.merge(pivot_median, on='uid', how='left')


# In[216]:


df_acc_test.shape


# In[217]:


pivot_sum = pd.pivot_table(df_acc_test, 
                            index='uid', 
                            columns='credit_type', 
                            values='loan_amount', 
                            aggfunc='sum', 
                            fill_value=0)

# Flatten the columns of the pivot table
pivot_sum.columns = [f'{credit}_total_loan' for credit in pivot_sum.columns]

# Reset the index to merge with the original dataframe
pivot_sum.reset_index(inplace=True)

df_acc_test = df_acc_test.merge(pivot_sum, on='uid', how='left')


# In[219]:


df_acc_test.head()


# In[218]:


df_acc_test.shape


# In[220]:


#Loan Status
df_acc_test['loan_status'] = df_acc_test['closed_date'].apply(lambda x: 'Open' if pd.isnull(x) else 'Closed')


# In[221]:


most_frequent_credit_type = df_acc_test.groupby('uid')['credit_type'].agg(lambda x: x.value_counts().idxmax())

df_acc_test['most_frequent_credit_type'] = df_acc_test['uid'].map(most_frequent_credit_type)


# ## Feature Selection

# In[222]:


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


df_acc_test = df_acc_test.drop(columns=columns_to_drop)


# In[223]:


df_acc_test.shape


# In[224]:


x = df_acc_test["uid"].value_counts()


# In[ ]:


x


# In[226]:


df_acc_test["uid"].nunique()


# In[ ]:


df_acc_test.duplicated(subset="uid").sum()


# In[227]:


df_acc_test = df_acc_test.drop_duplicates(subset="uid",keep="first")


# In[228]:


df_acc_test.shape


# In[229]:


acc_prefix = 'accList_'
df_acc_test = df_acc_test.rename(columns=lambda x: acc_prefix + x if x != 'uid' else x)


# In[230]:


user_dict = df_acc_test[df_acc_test["uid"]=='AAA14437029'].to_dict('records')


# In[231]:


user_dict


# In[233]:


df_acc_test.to_csv('df_acc_test.csv', index=False)


# In[ ]:





# In[ ]:




