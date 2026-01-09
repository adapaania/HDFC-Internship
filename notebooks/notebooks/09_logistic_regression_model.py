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

# In[ ]:


import pandas as pd


# # Load Dataset

# ##### TRAIN SET

# In[3]:


train = pd.read_csv("./data/df_train.csv")


# In[4]:


train = train.drop(columns='accList_loan_status')


# In[5]:


df_train = train.copy()


# In[6]:


df_train["TARGET"].value_counts()


# In[7]:


df_train["TARGET"].mean()


# In[8]:


og_flag = train[["uid","TARGET"]].copy()


# In[9]:


train.shape


# In[10]:


train.info()


# In[11]:


train[train.select_dtypes(include=['object']).columns]


# In[12]:


x = train.columns.to_list()
x


# ##### TEST SET

# In[13]:


test = pd.read_csv("/Users/aaniaadap/Desktop/HDFC Internship/df_test.csv")


# In[14]:


test.shape


# In[15]:


test_target=pd.read_csv('/Users/aaniaadap/Desktop/HDFC Internship/Monsoon Project/senior_ds_test/data/test/target.csv')


# In[16]:


test=test.merge(test_target,on='uid',how='inner')


# In[17]:


test = test.drop(columns="accList_loan_status")


# In[18]:


df_test = test.copy()


# In[19]:


test.shape


# In[20]:


test.info()


# In[21]:


test.select_dtypes(include=['object']).columns


# In[22]:


y = test.columns.to_list()
y


# In[23]:


set(x) - set(y)


# In[24]:


missing_cols = set(x) - set(y)
for col in missing_cols:
    test[col] = 0


# In[25]:


set(train.columns.to_list()) - set(test.columns.to_list())


# In[26]:


test.shape


# In[27]:


test["TARGET"].mean()


# 
# ## DISTRIBUTION

# In[28]:


train.describe()


# In[ ]:


test.describe()


# ## PREPROCESSING

# In[29]:


# Drop the uid column as it is likely not useful for model training
train = train.drop(columns=['uid'])
test = test.drop(columns=['uid'])


# ## Label Encoding

# In[30]:


from sklearn.preprocessing import LabelEncoder

categorical_columns = ['accList_most_frequent_credit_type']

# Combine train and test sets for fitting label encoder
combined_data = pd.concat([train[categorical_columns], test[categorical_columns]])

combined_data = combined_data.astype(str)

label_encoder = LabelEncoder()
for col in categorical_columns:
    label_encoder.fit(combined_data[col]) 
    train[col] = label_encoder.transform(train[col])
    test[col] = label_encoder.transform(test[col])


# ## Scaling the Data

# In[31]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical_cols = train.drop(columns=['TARGET']).columns
numerical_cols_test = test.drop(columns=["TARGET"]).columns

scaler.fit_transform(train[numerical_cols])

scaled_train = scaler.transform(train[numerical_cols])
scaled_test = scaler.transform(test[numerical_cols])

# Convert the scaled data back to a DataFrame
train_scaled = pd.DataFrame(scaled_train, columns=numerical_cols)
train_scaled['TARGET'] = train['TARGET']

test_scaled = pd.DataFrame(scaled_test, columns=numerical_cols)
test_scaled["TARGET"]=test["TARGET"]


# In[32]:


print("Scaled Training Data:")
train_scaled


# In[33]:


print("\nScaled Test Data:")
test_scaled


# # Feature Selection

# In[34]:


x = train_scaled.drop(columns='TARGET')
y = train_scaled["TARGET"]


# In[35]:


test_x = test_scaled.drop(columns='TARGET')
test_y = test_scaled["TARGET"]


# In[36]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[37]:


y_test.mean()*100


# In[38]:


from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_columns = ['accList_most_frequent_credit_type']
numerical_columns = x_train.columns.difference(categorical_columns)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', SimpleImputer(strategy='mean'), numerical_columns)
    ]
)
x_train = preprocessor.fit_transform(x_train)
x_test = preprocessor.transform(x_test)
test_x = preprocessor.transform(test_x)


# In[39]:


feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
feature_names_num = numerical_columns.tolist()

feature_names = np.concatenate([feature_names_cat, feature_names_num])

x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)
test_x_df = pd.DataFrame(test_x, columns=feature_names)


# In[40]:


print("Transformed Training DataFrame:")
(x_train_df.head())


# In[41]:


print("\nTransformed Test DataFrame:")
(x_test_df.head())


# In[42]:


print("\nOriginal Transformed Test DataFrame:")
(test_x_df.head())


# ## SMOTE

# In[45]:


pip install imblearn


# In[46]:


from imblearn.over_sampling import SMOTE


# In[54]:


smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train_df, y_train)


# ## RFE

# In[56]:


import lightgbm as lgb
from sklearn.feature_selection import RFE

lgb_model = lgb.LGBMClassifier()

rfe = RFE(estimator=lgb_model, n_features_to_select=50, step=5)

rfe.fit(x_train_smote, y_train_smote)


X_train_rfe = rfe.transform(x_train_smote)
X_test_rfe = rfe.transform(x_test_df)
test_x_rfe = rfe.transform(test_x_df)


# In[58]:


# Print selected features
selected_features = x_train_smote.columns[rfe.support_].tolist()
print("Selected Features:")
for feature in selected_features:
    print(f"- {feature}")

# Optionally, you can also get dropped features
dropped_features = x_train_df.columns[~rfe.support_].tolist()
print("\nDropped Features:")
for feature in dropped_features:
    print(f"- {feature}")


# In[59]:


feature_names = x_train_smote.columns

# Print feature rankings and names
print("Feature Rankings:")
for rank, name in sorted(zip(rfe.ranking_, feature_names)):
    print(f"Rank {rank}: {name}")


# In[60]:


feature_rankings = rfe.ranking_

# Get list of feature names
feature_names = x_train_smote.columns

# Create a DataFrame to store feature names and their ranks
feature_ranks_df = pd.DataFrame({
    'Feature': feature_names,
    'Rank': feature_rankings
})

# Filter features where rank is less than or equal to 5
selected_features_top5 = feature_ranks_df[feature_ranks_df['Rank'] <= 5]

# Sort by rank to maintain the order
selected_features_top5 = selected_features_top5.sort_values(by='Rank')

# Extract the feature names with ranks up to 5
selected_feature_names = selected_features_top5['Feature'].tolist()

# Display selected feature names and their ranks
print("Selected Features (up to Rank 5):")
selected_features_top5


# In[52]:


len(selected_feature_names)


# # MODELS

# In[61]:


x_train = x_train_smote[selected_feature_names]
x_test = x_test_df[selected_feature_names]
test_x = test_x_df[selected_feature_names]


# ## LOGISTIC REGRESSION

# In[62]:


from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression model
logreg_model = LogisticRegression(max_iter=2000, solver='saga')


# Fit the model on the training data
logreg_model.fit(x_train, y_train_smote)

# Calculate AUC score (assuming you have y_test and predicted probabilities)
from sklearn.metrics import roc_auc_score

y_pred = logreg_model.predict_proba(x_test)[:, 1]  # Assuming binary classification
auc_score = roc_auc_score(y_test, y_pred)

print("AUC score:", auc_score)


# ### Grid Search

# In[64]:


from sklearn.model_selection import GridSearchCV

params= {
    'C':[1,10,100],
         'max_iter':[1000,2000],
         'solver':["saga"]
}

clf = GridSearchCV(logreg_model, param_grid = params, cv=5, verbose = 3,scoring='roc_auc',return_train_score=False)

clf.fit(x_train, y_train_smote)


# In[65]:


pd.DataFrame(clf.cv_results_)


# In[66]:


print("Best parameters found: ", clf.best_params_)
print("Best cross-validation score: ", clf.best_score_)


# ##### Predict on Test Set and Calculating the Probability of the Outcome

# In[73]:


y_pred_train_lr = clf.best_estimator_.predict(x_train)
y_pred_test_lr = clf.best_estimator_.predict(x_test)
y_pred_og_test_lr = clf.best_estimator_.predict(test_x)

y_pred_train_lr_proba = clf.best_estimator_.predict_proba(x_train)
y_pred_test_lr_proba = clf.best_estimator_.predict_proba(x_test)
y_pred_og_test_proba_lr = clf.best_estimator_.predict_proba(test_x)


# In[74]:


print("Predicted class labels for training data:", y_pred_train_lr[:5])
print("Predicted probabilities for training data:\n", y_pred_train_lr_proba[:5])


print("Predicted class labels for test data:", y_pred_test_lr[:20])
print("Predicted probabilities for test data:\n", y_pred_test_lr_proba[:5])


print("Predicted class labels for test data:", y_pred_og_test_lr[:20])
print("Predicted probabilities for test data:\n", y_pred_og_test_proba_lr[:5])


# #### Feature Importances

# In[75]:


coefficients = logreg_model.coef_[0]

# Create a DataFrame to see feature importance
feature_importance_lr = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': coefficients
})

# Sort features by their absolute coefficient values
feature_importance_lr['Absolute_Coefficient'] = np.abs(feature_importance_lr['Coefficient'])
feature_importance_lr = feature_importance_lr.sort_values(by='Absolute_Coefficient', ascending=False)
feature_importance_lr.head(20)


# In[70]:


test_y=test["TARGET"]


# #### AUC-ROC SCORE

# In[80]:


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

y_train_prob_lr = y_pred_train_lr_proba[:, 1]
y_test_prob_lr = y_pred_test_lr_proba[:, 1]
y_pred_og_test_prob_lr = y_pred_og_test_proba_lr[:, 1]

train_auc_lr = roc_auc_score(y_train_smote, y_train_prob_lr)
test_auc_lr = roc_auc_score(y_test, y_test_prob_lr)
og_test_auc_lr = roc_auc_score(test_y,y_pred_og_test_prob_lr)

print(f"Train AUC-ROC: {train_auc_lr}")
print(f"Test AUC-ROC: {test_auc_lr}")
print(f"Original Test Dataset AUC-ROC: {og_test_auc_lr}")


# #### GINI COEFFICIENT

# In[81]:


# gini_coefficient = 2 * roc_auc - 1
gini_train = 2 * train_auc_lr - 1
gini_test =  2 * test_auc_lr - 1
gini_og_test = 2 * og_test_auc_lr - 1


# In[82]:


print(f"Gini Coefficient for Train: {gini_train}")
print(f"Gini Coefficient for Test Split: {gini_test}")
print(f"Gini Coefficient for Original Test Dataset: {gini_og_test}")


# #### KS Statistic

# In[84]:


def ks_statistic(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    data.sort_values(by='y_pred_proba', ascending=False, inplace=True)
    data['cum_event_rate'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['cum_non_event_rate'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    data['ks'] = np.abs(data['cum_event_rate'] - data['cum_non_event_rate'])
    ks_stat = data['ks'].max()
    return ks_stat

ks_stat_train = ks_statistic(y_train_smote,y_train_prob_lr)
ks_stat_test_split = ks_statistic(y_test,y_test_prob_lr)
ks_stat_test_og = ks_statistic(test_y,y_pred_og_test_prob_lr)


# In[85]:


print(f"KS Statistic for Train: {ks_stat_train:.4f}")
print(f"KS Statistic for Test Split: {ks_stat_test_split:.4f}")
print(f"KS Statistic for Original Test Dataset: {ks_stat_test_og:.4f}")


# # PLOTS

# ## AUC-ROC CURVE

# In[66]:


import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(test_y, y_pred_og_test_prob_lr)

# Step 3: Compute the AUC score
auc_score = roc_auc_score(test_y,y_pred_og_test_prob_lr)

# Step 4: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# In[67]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nTraining Set Evaluation")
print("Accuracy:", accuracy_score(y_train, y_pred_train_lr))
print("Classification Report:\n", classification_report(y_train, y_pred_train_lr))


# In[68]:


print("\nTest Set Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_test_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_test_lr))


# ## CONFUSION MATRIX

# In[69]:


import seaborn as sns
import matplotlib.pyplot as plt

cm_train = confusion_matrix(y_train, y_pred_train_lr)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Training Set')
plt.show()


cm_test = confusion_matrix(y_test, y_pred_test_lr)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.show()


# ## Decile Chart

# In[258]:


test_y_prob = logreg_model.predict_proba(test_x)[:, 1]
test_y = test_y.reset_index(drop=True)

df = pd.DataFrame({'actual': test_y, 'pred_prob': test_y_prob})


# Rank the data by predicted probabilities in descending order
df = df.sort_values(by='pred_prob', ascending=False).reset_index(drop=True)


df['decile'] = pd.qcut(df['pred_prob'], 10, labels=False)+1
df['decile'] = 11 - df['decile']

df


# In[259]:


gain = df.groupby('decile').agg(
    num_responses=('actual', 'sum'), 
    total=('actual', 'count'),
    # range_prob=('pred_prob', lambda x: (np.round(np.max(x),3), np.round(np.min(x),3)))
)

gain.sort_values(by="decile", inplace=True)

gain['response_rate'] = gain['num_responses'] / gain['total']
gain['response_rate_percentage'] = gain['response_rate']*100

gain['cumulative_response'] = gain['num_responses'].cumsum()

gain['cumulative_response_rate'] = gain['cumulative_response'] / gain['num_responses'].sum()

gain['lift'] = gain['cumulative_response_rate'] / (gain['total'].cumsum() / gain['total'].sum())

gain = gain.reset_index()
gain


# In[260]:


gain["response_rate_percentage"].mean()


# ### Gain Chart

# In[261]:


cum_response_rate = pd.concat([pd.Series([0]), 
                               gain['cumulative_response_rate']*100])

cum_rate = pd.concat([pd.Series([0]), (gain['total'].cumsum()/gain['total'].sum()*100)])


# In[262]:


plt.figure(figsize=(8,4),dpi=120)

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
sns.lineplot(x=(np.arange(0,1.1,.1)), y=cum_response_rate, marker='o',color='brown')
sns.lineplot(x=(np.arange(0,1.1,.1)), y=cum_rate, marker='o',color="orange")
plt.title('Gains Chart')
plt.xlabel('Deciles')
plt.ylabel('% of Cummulative Response Rate')

for x, y in zip(np.arange(0, 1.1, 0.1), cum_response_rate):
    plt.text(x, y, f'{y:.2f}%', ha='right', va='bottom', fontsize=8, color='brown',alpha=0.5)

for x, y in zip(np.arange(0, 1.1, 0.1), cum_rate):
    plt.text(x, y, f'{y:.2f}%', ha='right', va='top', fontsize=8, color='orange',alpha=0.5)

plt.show()


# ### Decile Chart

# In[438]:


# Response Rate Chart
plt.figure(figsize=(10,6))

average_response_rate = gain['response_rate_percentage'].mean()

for spine in plt.gca().spines.values():
    spine.set_visible(False)

sns.barplot(x='decile', y='response_rate_percentage', data=gain,palette='YlOrBr',alpha=0.7)

plt.axhline(y=average_response_rate, color='maroon', alpha=0.4,linestyle='--', label=f'Average: {average_response_rate:.2f}%')

plt.title('Decile Chart')
plt.xlabel('Deciles')
plt.ylabel('% of Correctly Predicted Bad Loans')
plt.show()


# In[264]:


gain['decile'].to_list()


# In[265]:


gain.lift[::-1]


# ### Lift Curve

# In[266]:


plt.figure(figsize=(10, 6),dpi=120)
sns.lineplot(x=gain['decile'].to_list(), y=gain.lift[::-1],marker='o')
plt.axhline(y=1, color='orange', linestyle='--')
plt.title('Lift Curve')
plt.ylim(0,7)
plt.xlabel('Deciles')
plt.ylabel('Lift')
plt.show()


# #### Deciles for features

# ##### Feature 1

# In[496]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, Formatter

# Assuming df_test, test_y, and feature_name are defined as per your context
feature_name = 'enqList_total_enquiries_per_customer'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df1 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df1.sort_values(by=feature_name)

df1['decile'] = pd.qcut(df1[feature_name], 10, labels=False, duplicates='drop')


decile_stats = df1.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df1.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df1['actual'].mean()

decile_limits = df1.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)
# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')


# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for Total Number of Enquiries made by a Borrower')
ax.set_xticks(decile_stats['decile'])
ax.set_xticklabels([f'{decile_limits.loc[d, "min"]:.2f} <= val \n<={decile_limits.loc[d, "max"]:.2f}' for d in decile_limits['decile']],rotation=45,ha='center')

# Show plot
plt.tight_layout()
plt.show()


# ##### Feature 2

# In[504]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df_test, test_y, and feature_name are defined as per your context
feature_name = 'accList_avg_loan_amount'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by='accList_avg_loan_amount')

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate

ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')


# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Average Amount of Loan Taken by a Borrower')
ax.set_xticks(decile_stats['decile'])

def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)


plt.tight_layout()
plt.show()


# ##### Feature 3

# In[501]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df_test, test_y, and feature_name are defined as per your context
feature_name = 'enqList_enquiries_last_year'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Number of Enquiries a Borrower has Made in the Past Year')
ax.set_xticks(decile_stats['decile'])

def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)

#ax.set_xticklabels([f'{decile_limits.loc[d, "min"]:.2f} <= val \n<= {decile_limits.loc[d, "max"]:.2f}' for d in decile_limits['decile']], rotation=45)

# Show plot
plt.tight_layout()
plt.show()


# ##### Feature 4

# In[505]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_name = 'accList_median_loan_amount'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Median Amount of Loan Taken by a Borrower')
ax.set_xticks(decile_stats['decile'])
def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)

# Show plot
plt.tight_layout()
plt.show()


# ##### Feature 5

# In[506]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_name = 'accList_min_loan_amt'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Minimum Amount of Loan Taken by a Borrower')
ax.set_xticks(decile_stats['decile'])
def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)
# Show plot
plt.tight_layout()
plt.show()


# ##### Feature 6

# In[512]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_name = 'enqList_enquiries_last_9_months'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Number of Enquiries a Borrower has Made in the Past 9 Months')
ax.set_xticks(decile_stats['decile'])
def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)

# Show plot
plt.tight_layout()
plt.show()


# ##### Feature 7

# In[508]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_name = 'accList_avg_ontime_payments'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Average Number of Ontime Payments Made by the Customer')
ax.set_xticks(decile_stats['decile'])
def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)
plt.tight_layout()
plt.show()


# ##### Feature 8

# In[513]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_name = 'accList_Credit card_total_loan'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Total Amount of Credit Card Loan Taken by a Borrower')
ax.set_xticks(decile_stats['decile'])
def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)

plt.tight_layout()
plt.show()


# ##### Feature 9

# In[546]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

feature_name = 'accList_Consumer credit_total_loan'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

df2.sort_values(by=feature_name)

df2['decile'] = pd.qcut(df2[feature_name], 10, labels=False, duplicates='drop')

decile_limits = df2.groupby('decile')[feature_name].agg(['min', 'max']).reset_index()

decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate the response rate for each decile
decile_stats = df2.groupby('decile').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats['decile'], decile_stats['bad_rate']*100, color=plt.cm.YlOrBr(np.linspace(1,0.5  ,10)),alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2,alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown',alpha=0.7, fontsize=12, va='center',ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title('Decile Graph for the Total Amount of Consumer Credit Loans Taken by a Borrower')
ax.set_xticks(decile_stats['decile'])
def format_tick_label(d):
    current_max = decile_limits.loc[d, "max"]
    next_min = decile_limits.loc[d+1, "min"] if d < len(decile_limits)-1 else np.inf  # Handle last decile case
    return f'{decile_limits.loc[d, "min"]:.2f} <= val \n< {next_min:.2f}'

ax.set_xticklabels([format_tick_label(d) for d in decile_limits.index], rotation=45)


plt.tight_layout()
plt.show()


# ##### Feature 10

# In[519]:


d = df2[df2["accList_Mortgage_total_loan"]!=0]


# In[520]:


d["accList_Mortgage_total_loan"].min()


# In[ ]:


#### Mortagage Fetaure - Outliers! 


# In[549]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming df_test, test_y, and feature_name are defined as per your context
feature_name = 'accList_Mortgage_total_loan'

# Reset index of test_y if needed
test_y = test_y.reset_index(drop=True)

# Create DataFrame with actual outcomes and the specific feature
df2 = pd.DataFrame({'actual': test_y, feature_name: df_test[feature_name]})

# Sort the DataFrame by the feature_name
df2 = df2.sort_values(by=feature_name)

# Define mortgage total loan amount categories and bins
category_labels = [f'{i*500000} - {(i+1)*5000000}' for i in range(10)]
category_bins = [i*500000 for i in range(11)]

# Assign categories based on mortgage total loan amount
df2['category'] = pd.cut(df2[feature_name], bins=category_bins, labels=category_labels, right=False)

# Calculate decile statistics within each category
decile_stats = df2.groupby('category').agg(
    count=('actual', 'size'),
    bad_rate=('actual', 'mean')
).reset_index()

# Calculate overall portfolio bad rate
portfolio_bad_rate = df2['actual'].mean()

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.bar(decile_stats.index, decile_stats['bad_rate']*100, 
              color=plt.cm.YlOrBr(np.linspace(1, 0.5, len(category_labels))), alpha=0.8)

# Adding text for each bar
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom')

# Add horizontal line for portfolio bad rate
ax.axhline(y=portfolio_bad_rate*100, color='brown', linestyle='--', linewidth=2, alpha=0.5)
ax.text(len(decile_stats)-0.5, portfolio_bad_rate*100, f'Portfolio\nBad Rate\nis {portfolio_bad_rate*100:.2f}%', 
        color='brown', alpha=0.7, fontsize=12, va='center', ha='center')

# Set labels and title
ax.set_xlabel('Deciles')
ax.set_ylabel('Bad Rate (%)')
ax.set_title(' Total Amount of Mortgage Loans a Borrower has Taken')

# Set x-axis ticks and labels
ax.set_xticks(range(len(category_labels)))
ax.set_xticklabels(category_labels, rotation=45)

# Show plot
plt.tight_layout()
plt.show()


# In[543]:


df[df2["category"].isna()].sort_values(by='feature')


# In[ ]:




