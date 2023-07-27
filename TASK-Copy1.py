#!/usr/bin/env python
# coding: utf-8

# # Iris Flowers Classification ML Project 

# In[1]:


# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt 


# In[2]:


# Load the Iris dataset 
from sklearn.datasets import load_iris
iris = load_iris()


# In[3]:


df=pd.read_csv("iris 1.csv")


# In[4]:


df.head()


# In[5]:


# Create a DataFrame from the iris dataset
data = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])


# In[6]:


# Split the data into features (X) and labels (y)
X = data.drop('target', axis=1)
y = data['target']


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[9]:


# Train the K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)


# In[10]:


# Make predictions on the test set
y_pred = knn.predict(X_test_scaled)


# In[11]:


# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[12]:


# Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[13]:


# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:





# In[ ]:




