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

# In[1]:


import pandas as pd


# # Load Dataset

# ##### TRAIN SET

# In[2]:


train = pd.read_csv("./data/df_train.csv")


# In[3]:


train = train.drop(columns='accList_loan_status')


# In[4]:


df_train = train.copy()


# In[5]:


df_train["TARGET"].value_counts()


# In[6]:


og_flag = train[["uid","TARGET"]].copy()


# In[7]:


train.shape


# In[8]:


train.info()


# In[9]:


train[train.select_dtypes(include=['object']).columns]


# In[10]:


x = train.columns.to_list()
x


# ##### TEST SET

# In[11]:


test = pd.read_csv("/Users/aaniaadap/Desktop/HDFC Internship/df_test.csv")


# In[12]:


test.shape


# In[13]:


test_target=pd.read_csv('/Users/aaniaadap/Desktop/HDFC Internship/Monsoon Project/senior_ds_test/data/test/target.csv')


# In[14]:


test=test.merge(test_target,on='uid',how='inner')


# In[15]:


test = test.drop(columns="accList_loan_status")


# In[16]:


df_test = test.copy()


# In[17]:


test.shape


# In[18]:


test.info()


# In[19]:


test.select_dtypes(include=['object']).columns


# In[20]:


y = test.columns.to_list()
y


# In[21]:


set(x) - set(y)


# In[22]:


missing_cols = set(x) - set(y)
for col in missing_cols:
    test[col] = 0


# In[33]:


set(train.columns.to_list()) - set(test.columns.to_list())


# In[34]:


test.shape


# 
# ## DISTRIBUTION

# In[35]:


train.describe()


# In[ ]:


test.describe()


# ## PREPROCESSING

# In[36]:


# Drop the uid column as it is likely not useful for model training
train = train.drop(columns=['uid'])
test = test.drop(columns=['uid'])


# In[37]:


train[['accList_most_frequent_credit_type']]


# In[38]:


test[['accList_most_frequent_credit_type']]


# ## Label Encoding

# In[ ]:


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

# In[42]:


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


# In[43]:


print("Scaled Training Data:")
train_scaled


# In[44]:


print("\nScaled Test Data:")
test_scaled


# # Feature Selection

# In[45]:


x = train_scaled.drop(columns='TARGET')
y = train_scaled["TARGET"]


# In[46]:


test_x = test_scaled.drop(columns='TARGET')
test_y = test_scaled["TARGET"]


# In[47]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)


# In[48]:


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


# In[50]:


feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
feature_names_num = numerical_columns.tolist()

feature_names = np.concatenate([feature_names_cat, feature_names_num])

x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)
test_x_df = pd.DataFrame(test_x, columns=feature_names)


# In[51]:


print("Transformed Training DataFrame:")
(x_train_df.head())


# In[52]:


print("\nTransformed Test DataFrame:")
(x_test_df.head())


# In[53]:


print("\nOriginal Transformed Test DataFrame:")
(test_x_df.head())


# ## RFE

# In[54]:


import lightgbm as lgb
from sklearn.feature_selection import RFE

lgb_model = lgb.LGBMClassifier()

rfe = RFE(estimator=lgb_model, n_features_to_select=50, step=5)

rfe.fit(x_train_df, y_train)


X_train_rfe = rfe.transform(x_train_df)
X_test_rfe = rfe.transform(x_test_df)
test_x_rfe = rfe.transform(test_x_df)


# In[55]:


# Print selected features
selected_features = x_train_df.columns[rfe.support_].tolist()
print("Selected Features:")
for feature in selected_features:
    print(f"- {feature}")

# Optionally, you can also get dropped features
dropped_features = x_train_df.columns[~rfe.support_].tolist()
print("\nDropped Features:")
for feature in dropped_features:
    print(f"- {feature}")


# In[56]:


feature_names = x_train_df.columns

# Print feature rankings and names
print("Feature Rankings:")
for rank, name in sorted(zip(rfe.ranking_, feature_names)):
    print(f"Rank {rank}: {name}")


# In[57]:


feature_rankings = rfe.ranking_

# Get list of feature names
feature_names = x_train_df.columns

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


# In[58]:


len(selected_feature_names)


# # MODELS

# In[59]:


x_train = x_train_df[selected_feature_names]
x_test = x_test_df[selected_feature_names]
test_x = test_x_df[selected_feature_names]


# ## LGBM

# In[ ]:


import lightgbm as lgb


# In[60]:


lgb_model = lgb_model = lgb.LGBMClassifier(
    objective='binary',         
    metric='auc',               
    random_state=42             
)


# ### Grid Search CV

# In[66]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'learning_rate': [0.005, 0.001],
    'n_estimators': [200, 300],
    'num_leaves': [20, 30],
    'max_depth': [10, 20],
    'reg_lambda': [1, 10],
    'reg_alpha': [1, 10],
}

grid_search_lgbm = GridSearchCV(estimator=lgb_model, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=3, n_jobs=-1)
grid_search_lgbm.fit(x_train, y_train)

pd.DataFrame(grid_search_lgbm.cv_results_)


# In[67]:


pd.DataFrame(grid_search_lgbm.cv_results_)


# ##### IMPORTANT FEATURES

# In[96]:


feature_importance_lgb = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': lgb_model.feature_importances_
})

# Sort features by importance
feature_importance_lgb = feature_importance_lgb.sort_values(by='Importance', ascending=False)

print(feature_importance_lgb)


# ##### Prediction Info

# In[68]:


y_pred_train_lgbm = grid_search_lgbm.best_estimator_.predict(x_train)
y_pred_test_lgbm = grid_search_lgbm.best_estimator_.predict(x_test)
y_pred_og_test_lgbm = grid_search_lgbm.best_estimator_.predict(test_x)

y_pred_train_lgbm_proba = grid_search_lgbm.best_estimator_.predict_proba(x_train)
y_pred_test_lgbm_proba = grid_search_lgbm.best_estimator_.predict_proba(x_test)
y_pred_og_test_lgbm_proba = grid_search_lgbm.best_estimator_.predict_proba(test_x)

print("Predicted class labels for training data:", y_pred_train_lgbm[:5])
print("Predicted probabilities for training data:\n", y_pred_train_lgbm_proba[:5])

print("Predicted class labels for test data:", y_pred_og_test_lgbm[:5])
print("Predicted probabilities for test data:\n", y_pred_og_test_lgbm_proba[:5])


print("Predicted class labels for test data:", y_pred_test_lgbm[:5])

print("Predicted probabilities for test data:\n", y_pred_test_lgbm_proba[:5])


# In[69]:


best_params = grid_search_lgbm.best_params_
best_score = grid_search_lgbm.best_score_


# In[70]:


print("Best Parameters:", best_params)
print("Best ROC-AUC Score:", best_score)


# ##### AUC_ROC SCORE

# In[72]:


from sklearn.metrics import roc_auc_score

y_train_prob_lgb = y_pred_train_lgbm_proba[:, 1]
y_test_prob_lgb = y_pred_test_lgbm_proba[:, 1]
og_y_test_prob_lgbm = y_pred_og_test_lgbm_proba[:,1]

train_auc_lgbm = roc_auc_score(y_train, y_train_prob_lgb)
test_auc_lgbm = roc_auc_score(y_test, y_test_prob_lgb)
og_test_auc_lgbm = roc_auc_score(test_y,og_y_test_prob_lgbm )

print(f"Train AUC-ROC: {train_auc_lgbm}")
print(f"Test AUC-ROC: {test_auc_lgbm}")
print(f"Original Test Dataset AUC-ROC: {og_test_auc_lgbm}")


# ##### GINI COEFFICIENT

# In[97]:


# gini_coefficient = 2 * roc_auc - 1
gini_train = 2 * train_auc_lgbm - 1
gini_test =  2 * test_auc_lgbm - 1
gini_og_test = 2 * og_test_auc_lgbm - 1


# In[98]:


print(f"Gini Coefficient for Train: {gini_train}")
print(f"Gini Coefficient for Test Split: {gini_test}")
print(f"Gini Coefficient for Original Test Dataset: {gini_og_test}")


# ##### KS Statistic

# In[101]:


def ks_statistic(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    data.sort_values(by='y_pred_proba', ascending=False, inplace=True)
    data['cum_event_rate'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['cum_non_event_rate'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    data['ks'] = np.abs(data['cum_event_rate'] - data['cum_non_event_rate'])
    ks_stat = data['ks'].max()
    return ks_stat

ks_stat_train = ks_statistic(y_train,y_train_prob_lgb)
ks_stat_test_split = ks_statistic(y_test,y_test_prob_lgb)
ks_stat_test_og = ks_statistic(test_y,og_y_test_prob_lgbm)


# In[102]:


print(f"KS Statistic for Train: {ks_stat_train:.4f}")
print(f"KS Statistic for Test Split: {ks_stat_test_split:.4f}")
print(f"KS Statistic for Original Test Dataset: {ks_stat_test_og:.4f}")


# In[ ]:





# # PLOTS 

# ## AUC-ROC 

# In[75]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob_lgb)

# Step 3: Compute the AUC score
auc_score = roc_auc_score(y_test, y_test_prob_lgb)

# Step 4: Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for LGBM')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# ## CONFUSION MATRIX

# In[77]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(y_train, y_pred_train_lgbm)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Training Set(LGBM)')
plt.show()


cm_test = confusion_matrix(y_test, y_pred_test_lgbm)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set(LGBM)')
plt.show()


# In[ ]:


Best Parameters: {'learning_rate': 0.01, 'max_depth': 15, 'n_estimators': 300, 'num_leaves': 30}
Train AUC-ROC: 0.6981864819067345
Test AUC-ROC: 0.656668514830222
    
Best Parameters: {'learning_rate': 0.005, 'max_depth': 20, 'n_estimators': 300, 'num_leaves': 30, 'reg_alpha': 1, 'reg_lambda': 1}
Train AUC-ROC: 0.666686695668577
Test AUC-ROC: 0.6427436607473809


# ## DECILE

# In[86]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming logreg_model, test_x, and test_y are already defined
test_y_prob = lgb_model.predict_proba(test_x)[:, 1]
test_y = test_y.reset_index(drop=True)

# Create a DataFrame with predicted probabilities and actual labels
df = pd.DataFrame({
    'predicted_prob': og_y_test_prob_lgbm,
    'actual': test_y
})

# Rank the data by predicted probabilities in descending order
df = df.sort_values(by='predicted_prob', ascending=False).reset_index(drop=True)

# Calculate decile groups (1 to 10)
df['decile'] = pd.qcut(df.index, 10, labels=False) + 1  # 1-based deciles


# In[87]:


df['avg_baseline'] = df['actual'].mean()


# In[88]:


avg_decile = df.groupby('decile')['actual'].mean()

# Merge the calculated avg_decile back to the original DataFrame df
df = df.merge(avg_decile, on='decile', suffixes=('', '_decile'))

# Rename the column to 'avg_decile' for clarity
df.rename(columns={'actual_decile': 'avg_decile'}, inplace=True)



# In[89]:


df['dec_lift'] = df['avg_decile'] / df['avg_baseline']


# In[93]:


df_sorted = df.sort_values(by='decile')


# In[94]:


df_sorted


# In[95]:


plt.figure(figsize=(10, 6))
plt.bar(df['decile'].unique(), df.groupby('decile')['dec_lift'].mean(), color='brown',alpha =0.5)
plt.xlabel('Deciles')
plt.ylabel('Decile-wise Lift Ratio')
plt.title('Decile-wise Lift Ratio')
plt.show()



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




