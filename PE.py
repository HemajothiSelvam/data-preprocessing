#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[2]:


data=pd.read_csv("user_data.csv")
data


# In[5]:



df = pd.DataFrame(data)

df.drop(columns=['UserID'], inplace=True)
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender']) 
df['HighSalary'] = (df['EstimatedSalary'] > 40000).astype(int)
plt.figure(figsize=(10, 5))
sns.countplot(x='HighSalary', hue='Gender', data=df)
plt.title('Salary Distribution by Gender')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='Age', y='EstimatedSalary', hue='HighSalary', data=df)
plt.title('Age vs Estimated Salary')
plt.show()
X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['HighSalary']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))


# In[ ]:




