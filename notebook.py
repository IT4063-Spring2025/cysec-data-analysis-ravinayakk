#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[2]:


link = "./Data/CySecData.csv"
df = pd.read_csv(link)



# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[3]:


df.head()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[4]:


df.describe()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[5]:


categorical_cols = ['protocol_type', 'service', 'flag']

dummy_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

dummy_df.head()


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[6]:


df_features = dummy_df.drop('class', axis=1)

df_features.head()


# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[7]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[8]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df_features)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[9]:


from sklearn.preprocessing import StandardScaler

X = dummy_df.drop('class', axis=1)

y = dummy_df['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[10]:


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# # Step 11: Defining the models
# TODO: Define the models to be evaluated.

# In[11]:


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(),
}


# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.

# In[12]:


from sklearn.model_selection import cross_val_score
import numpy as np

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=10, scoring='accuracy')
    print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std Dev = {scores.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:




