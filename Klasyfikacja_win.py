#!/usr/bin/env python
# coding: utf-8

# # Klasyfikacja jakości wina na podstawie testów fizykochemicznych
# ### https://archive.ics.uci.edu/ml/datasets/wine+quality

# In[60]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge, LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt


# In[28]:


df = pd.read_csv("winequality-red.csv", delimiter=';')

print(df)


# In[21]:


fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax=ax)
plt.show()


# In[18]:


sns.pairplot(
    df.iloc[:, :],
    hue="quality",
    corner=True
)
plt.show()


# In[29]:


y = df.pop('quality').values
X = df.to_numpy()


# #### Liczebność klas

# In[82]:


unique_elements, counts_elements = np.unique(y, return_counts=True)
df_tmp = pd.DataFrame([[k,v] for k, v in zip(unique_elements, counts_elements)])
df_tmp = df_tmp.set_index(0)
print(df_tmp)


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)


# ### Regresja logistyczna

# In[ ]:


reg = LinearRegression().fit(X_train, y_train)


# #### Współczynnik $R^2$

# Zbiór treningowy

# In[40]:


print(reg.score(X_train, y_train))


# Zbiór testowy

# In[41]:


print(reg.score(X_test, y_test))


# $y = b + a_1x_1 + a_2x_2 + ... $

# In[42]:


display(reg.coef_)
display(reg.intercept_)


# ### Regresja Bayesowska

# In[46]:


reg = BayesianRidge().fit(X_train, y_train)


# #### Współczynnik $R^2$

# Zbiór treningowy

# In[47]:


print(reg.score(X_train, y_train))


# Zbiór testowy

# In[48]:


print(reg.score(X_test, y_test))


# $y = b + a_1x_1 + a_2x_2 + ... $

# In[49]:


display(reg.coef_)
display(reg.intercept_)


# ### Regresja logistyczna

# In[57]:


scaler = preprocessing.StandardScaler().fit(X_train)

softmax_reg = LogisticRegression(multi_class="multinomial").fit(scaler.transform(X_train), y_train)


# #### Skuteczność modelu

# Zbiór treningowy

# In[58]:


print(softmax_reg.score(scaler.transform(X_train), y_train))


# Zbiór testowy

# In[61]:


print(softmax_reg.score(scaler.transform(X_test), y_test))


# ### Macierz pomyłek

# ##### Zbiór treningowy

# In[70]:


labels = [num for num in np.unique(y)]


# In[72]:


y_true = y_train
y_pred = softmax_reg.predict(scaler.transform(X_train))
cm = confusion_matrix(y_true, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=labels,
                  columns=labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)
plt.show()

# ##### Zbiór testowy

# In[83]:


y_true = y_test
y_pred = softmax_reg.predict(scaler.transform(X_test))
cm = confusion_matrix(y_true, y_pred, labels=labels)
df_cm = pd.DataFrame(cm, index=labels,
                  columns=labels)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True)
plt.show()


# In[ ]:




