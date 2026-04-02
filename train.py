#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,label_binarize
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix,ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier




# In[35]:


import warnings
warnings.filterwarnings("ignore")


# In[36]:


df = pd.read_csv("weather_classification_data.csv")


# In[37]:


df.head()


# In[38]:


df.info()



# In[39]:


df.describe()


# In[40]:


sns.countplot(x='Weather Type', data=df)
plt.title("Weather Type Distribution")
plt.show()





# In[42]:


cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    df[col] = df[col].str.lower().str.strip()


# In[43]:


df['temp_humidity'] = df['Temperature'] * df['Humidity']
df['wind_pressure'] = df['Wind Speed'] * df['Atmospheric Pressure']


# In[44]:


X = df.drop(columns=['Weather Type'])
y = df['Weather Type']

le = LabelEncoder()
y = le.fit_transform(y)


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# In[47]:


num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns


# In[48]:


X[num_cols].hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()


# In[49]:


sns.boxplot(data=X_train[num_cols])
plt.xticks(rotation=45)
plt.yscale('log')
plt.title("Before IQR (Original Data)") 
plt.show()


# In[50]:


for col in num_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

   
    X_train.loc[:, col] = X_train[col].clip(lower, upper)
    X_test.loc[:, col] = X_test[col].clip(lower, upper)


# In[51]:


sns.boxplot(data=X_train[num_cols])
plt.xticks(rotation=45)
plt.yscale('log')
plt.title("After IQR (Clipped Data)")

plt.show()


# In[52]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ]
)


# In[53]:


sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[60]:

pipeline = Pipeline([
    ('preprocessor', preprocessor),   
    ('feature_selection', SelectKBest(score_func=f_classif, k=11)),

    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        max_features='log2',
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)


# In[61]:


feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

print(feature_names)


# In[64]:


selector = pipeline.named_steps['feature_selection']
scores = selector.scores_

plt.figure(figsize=(10,5))
plt.bar(range(len(scores)), scores)
plt.title("Feature Importance (ANOVA F-score)")
plt.xlabel("Feature Index")
plt.ylabel("Score")
plt.show()


# In[68]:


train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipeline,  
    X=X_train,
    y=y_train,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)


# In[69]:


train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[70]:




plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10,6))

plt.plot(train_sizes, train_mean, 'o-', label="Training Score", linewidth=2)
plt.plot(train_sizes, test_mean, 'o-', label="Validation Score", linewidth=2)

plt.fill_between(train_sizes, 
                 train_mean - train_std, 
                 train_mean + train_std, 
                 alpha=0.1)

plt.fill_between(train_sizes, 
                 test_mean - test_std, 
                 test_mean + test_std, 
                 alpha=0.1)

desired_score = 0.95   
plt.axhline(y=desired_score, color='red', linestyle='--', 
            label=f"Desired Score ({desired_score})")

plt.xlabel("Training Size")
plt.ylabel("F1 Score")
plt.title("Learning Curve")

plt.legend(loc="best")
plt.grid(alpha=0.3)

plt.show()



y_pred = pipeline.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

classes = np.unique(y)
y_test_bin = label_binarize(y_test, classes=classes)

y_score = pipeline.predict_proba(X_test)

fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 6))

plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle='--', color='black')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend(loc='lower right')
plt.grid(alpha=0.3)

plt.show()


with open("weather_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model saved successfully!")
