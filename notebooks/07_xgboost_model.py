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


# In[ ]:


set(train.columns.to_list()) - set(test.columns.to_list())


# In[23]:


test.shape


# 
# ## DISTRIBUTION

# In[24]:


train.describe()


# In[ ]:


test.describe()


# ## PREPROCESSING

# In[25]:


# Drop the uid column as it is likely not useful for model training
train = train.drop(columns=['uid'])
test = test.drop(columns=['uid'])


# In[26]:


train[['accList_most_frequent_credit_type']]


# In[27]:


test[['accList_most_frequent_credit_type']]


# ## Label Encoding

# In[28]:


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


# In[40]:


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


# In[41]:


feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_columns)
feature_names_num = numerical_columns.tolist()

feature_names = np.concatenate([feature_names_cat, feature_names_num])

x_train_df = pd.DataFrame(x_train, columns=feature_names)
x_test_df = pd.DataFrame(x_test, columns=feature_names)
test_x_df = pd.DataFrame(test_x, columns=feature_names)


# In[42]:


print("Transformed Training DataFrame:")
(x_train_df.head())


# In[43]:


print("\nTransformed Test DataFrame:")
(x_test_df.head())


# In[44]:


print("\nOriginal Transformed Test DataFrame:")
(test_x_df.head())


# ## RFE

# In[45]:


import lightgbm as lgb
from sklearn.feature_selection import RFE

lgb_model = lgb.LGBMClassifier()

rfe = RFE(estimator=lgb_model, n_features_to_select=50, step=5)

rfe.fit(x_train_df, y_train)


X_train_rfe = rfe.transform(x_train_df)
X_test_rfe = rfe.transform(x_test_df)
test_x_rfe = rfe.transform(test_x_df)


# In[46]:


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


# In[47]:


feature_names = x_train_df.columns

# Print feature rankings and names
print("Feature Rankings:")
for rank, name in sorted(zip(rfe.ranking_, feature_names)):
    print(f"Rank {rank}: {name}")


# In[48]:


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


# In[49]:


len(selected_feature_names)


# # MODELS

# In[50]:


x_train = x_train_df[selected_feature_names]
x_test = x_test_df[selected_feature_names]
test_x = test_x_df[selected_feature_names]


# ## XGB

# ### Grid Search

# In[137]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

# Define a smaller parameter grid for quicker execution
param_grid = {
    'n_estimators': [150], 
    'learning_rate': [0.01],  
    'max_depth': [4],  
    'reg_alpha': [0, 0.1],  
    'reg_lambda': [1, 1.5]
}

# Instantiate the XGBClassifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='auc')

# Instantiate the GridSearchCV object
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=3, n_jobs=-1)

# Fit the model
grid_search_xgb.fit(x_train, y_train)

"""Train AUC-ROC: 0.6353003231919462
Test AUC-ROC: 0.6274422775845294
Original Test Dataset AUC-ROC: 0.6235649410452768"""


# In[91]:


pd.DataFrame(grid_search_xgb.cv_results_)


# In[138]:


print("Best parameters found: ", grid_search_xgb.best_params_)
print("Best cross-validation score: ", grid_search_xgb.best_score_)


# In[139]:


y_pred_train = grid_search_xgb.best_estimator_.predict(x_train)
y_pred_test = grid_search_xgb.best_estimator_.predict(x_test)
y_pred_og_test = grid_search_xgb.best_estimator_.predict(test_x)

y_pred_train_proba = grid_search_xgb.best_estimator_.predict_proba(x_train)
y_pred_test_proba = grid_search_xgb.best_estimator_.predict_proba(x_test)
y_pred_og_test_proba = grid_search_xgb.best_estimator_.predict_proba(test_x)


# In[ ]:


print("Predicted class labels for training data:", y_pred_train[:5])
print("Predicted probabilities for training data:\n", y_pred_train_proba[:5])


print("Predicted class labels for test data:", y_pred_test[:20])
print("Predicted probabilities for test data:\n", y_pred_test_proba[:5])


print("Predicted class labels for test data:", y_pred_og_test[:20])
print("Predicted probabilities for test data:\n", y_pred_og_test_proba[:5])


# In[140]:


coefficients = xgb_model.coef_[0]

# Create a DataFrame to see feature importance
feature_importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': coefficients
})

# Sort features by their absolute coefficient values
feature_importance['Absolute_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values(by='Absolute_Coefficient', ascending=False)
feature_importance.head(10)


# In[141]:


test_y=test["TARGET"]


# ##### AUC-ROC SCORE

# In[142]:


from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

y_train_prob = y_pred_train_proba[:, 1]
y_test_prob = y_pred_test_proba[:, 1]
y_pred_og_test_prob = y_pred_og_test_proba[:, 1]

train_auc = roc_auc_score(y_train, y_train_prob)
test_auc = roc_auc_score(y_test, y_test_prob)
og_test_auc = roc_auc_score(test_y,y_pred_og_test_prob)

print(f"Train AUC-ROC: {train_auc}")
print(f"Test AUC-ROC: {test_auc}")
print(f"Original Test Dataset AUC-ROC: {og_test_auc}")


# ##### GINI COEFFICIENT

# In[98]:


# gini_coefficient = 2 * roc_auc - 1
gini_train = 2 * train_auc - 1
gini_test =  2 * test_auc - 1
gini_og_test = 2 * og_test_auc - 1


# In[99]:


print(f"Gini Coefficient for Train: {gini_train}")
print(f"Gini Coefficient for Test Split: {gini_test}")
print(f"Gini Coefficient for Original Test Dataset: {gini_og_test}")


# ##### KS Statistic

# In[100]:


def ks_statistic(y_true, y_pred_proba):
    data = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
    data.sort_values(by='y_pred_proba', ascending=False, inplace=True)
    data['cum_event_rate'] = data['y_true'].cumsum() / data['y_true'].sum()
    data['cum_non_event_rate'] = (1 - data['y_true']).cumsum() / (1 - data['y_true']).sum()
    data['ks'] = np.abs(data['cum_event_rate'] - data['cum_non_event_rate'])
    ks_stat = data['ks'].max()
    return ks_stat

ks_stat_train = ks_statistic(y_train,y_train_prob)
ks_stat_test_split = ks_statistic(y_test,y_test_prob)
ks_stat_test_og = ks_statistic(test_y,y_pred_og_test_prob)


# In[101]:


print(f"KS Statistic for Train: {ks_stat_train:.4f}")
print(f"KS Statistic for Test Split: {ks_stat_test_split:.4f}")
print(f"KS Statistic for Original Test Dataset: {ks_stat_test_og:.4f}")


# # PLOTS

# ## AUC-ROC CURVE

# In[102]:


import matplotlib.pyplot as plt
fpr, tpr, thresholds = roc_curve(test_y, y_pred_og_test_prob)

# Step 3: Compute the AUC score
auc_score = roc_auc_score(test_y,y_pred_og_test_prob)

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


# In[103]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\nTraining Set Evaluation")
print("Accuracy:", accuracy_score(y_train, y_pred_train))
print("Classification Report:\n", classification_report(y_train, y_pred_train))


# In[104]:


print("\nTest Set Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Classification Report:\n", classification_report(y_test, y_pred_test))


# ## CONFUSION MATRIX

# In[105]:


import seaborn as sns
import matplotlib.pyplot as plt

cm_train = confusion_matrix(y_train, y_pred_train)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Training Set')
plt.show()


cm_test = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Set')
plt.show()


# ## Decile Chart

# In[109]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming logreg_model, test_x, and test_y are already defined
test_y_prob = y_pred_og_test_prob
test_y = test_y.reset_index(drop=True)

# Create a DataFrame with predicted probabilities and actual labels
df = pd.DataFrame({
    'predicted_prob': test_y_prob,
    'actual': test_y
})

# Rank the data by predicted probabilities in descending order
df = df.sort_values(by='predicted_prob', ascending=False).reset_index(drop=True)

# Calculate decile groups (1 to 10)
df['decile'] = pd.qcut(df.index, 10, labels=False) + 1  # 1-based deciles


# In[110]:


df


# In[111]:


df['avg_baseline'] = df['actual'].mean()


# In[112]:


avg_decile = df.groupby('decile')['actual'].mean()

# Merge the calculated avg_decile back to the original DataFrame df
df = df.merge(avg_decile, on='decile', suffixes=('', '_decile'))

# Rename the column to 'avg_decile' for clarity
df.rename(columns={'actual_decile': 'avg_decile'}, inplace=True)



# In[113]:


df['dec_lift'] = df['avg_decile'] / df['avg_baseline']


# In[114]:


df_sorted = df.sort_values(by='decile')


# In[115]:


df_sorted


# In[117]:


plt.figure(figsize=(10, 6))
plt.bar(df['decile'].unique(), df.groupby('decile')['dec_lift'].mean(), color='brown',alpha =0.7)
plt.xlabel('Deciles')
plt.ylabel('Decile-wise Lift Ratio')
plt.title('Decile-wise Lift Ratio')
plt.show()


# In[ ]:





# ##### The model performs well in predicting the majority class (class 0) with a high number of True Negatives (135172) and low False Positives (13).
# ##### However, the model struggles to identify the minority class (class 1). There are a significant number of False Negatives (321), indicating the model missed many instances of class 1.

# In[ ]:




