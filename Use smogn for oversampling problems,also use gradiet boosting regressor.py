#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[9]:


get_ipython().system('pip install smogn')


# In[10]:


import smogn


# In[11]:


df=pd.read_csv("D:\\data\\day.csv")


# In[12]:


df.head()


# In[13]:


plt.figure(figsize=(14,10))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[14]:


corr = df.corr()
c1 = corr.abs().unstack()
c1.sort_values(ascending = False)[15:27:2]


# In[15]:


cate_cols = ["dteday", "season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
for col in cate_cols:
    df[col] = df[col].astype('category')


# In[16]:


print(df.info())


# In[17]:


df = df.rename(columns= {'dteday':'date', 'yr':'year', 'mnth':'month', 'weathersit': 'weather', 'hum':'humidity', 'cnt':'count'})
df.head()


# In[18]:


df=df.drop(columns=['instant', 'atemp', 'date', 'year'],axis=1)


# In[19]:


categorycols=['season', 'month', 'weekday', 'weather','workingday','holiday']
df = pd.get_dummies(df, columns = categorycols,drop_first=True)
df.head()


# In[20]:


dt = smogn.smoter(
                data=df,
                y='count',
                k=5,
                samp_method='extreme',
                rel_thres=0.9,
                rel_method='auto',
                rel_xtrm_type='high',
                rel_coef=0.9
            )


# In[21]:


dt.head()


# In[23]:


X = dt.drop(['casual', 'registered', 'count'], axis=1)
y = dt[['casual', 'registered']]


# In[24]:


X.shape, y.shape


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error
from sklearn.ensemble import  RandomForestRegressor


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[27]:


model_rf = RandomForestRegressor(random_state=42)


# In[28]:


model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)


# In[29]:


print('Testing R2 Score: ', r2_score(y_test, pred_rf)*100)
print('Testing RMSE: ', np.sqrt(mean_squared_error(y_test, pred_rf)))
print('Testing MAE: ', mean_absolute_error(y_test, pred_rf))
print('Testing MSE: ', mean_squared_error(y_test, pred_rf))


# In[30]:


pred_rf_trn = model_rf.predict(X_train)


# In[31]:


print('Training R2 Score: ', r2_score(y_train, pred_rf_trn)*100)
print('Training RMSE: ', np.sqrt(mean_squared_error(y_train, pred_rf_trn)))
print('Training MAE: ', mean_absolute_error(y_train, pred_rf_trn))
print('Training MSE: ', mean_squared_error(y_train, pred_rf_trn))


# In[42]:


#Gradient boosting Regressor


# In[36]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[37]:


model_gb = MultiOutputRegressor(GradientBoostingRegressor(random_state=2))

model_gb.fit(X_train, y_train)
pred_gb = model_gb.predict(X_test)


# In[38]:


print('Testing R2 Score: ', r2_score(y_test, pred_gb)*100)
print('Testing RMSE: ', np.sqrt(mean_squared_error(y_test, pred_gb)))
print('Testing MAE: ', mean_absolute_error(y_test, pred_gb))
print('Testing MSE: ', mean_squared_error(y_test, pred_gb))


# In[39]:


pred_gb[:4]


# In[40]:


pred_gb_trn = model_gb.predict(X_train)


# In[41]:


print('Training R2 Score: ', r2_score(y_train, pred_gb_trn)*100)
print('Training RMSE: ', np.sqrt(mean_squared_error(y_train, pred_gb_trn)))
print('Training MAE: ', mean_absolute_error(y_train, pred_gb_trn))
print('Training MSE: ', mean_squared_error(y_train, pred_gb_trn))


# In[ ]:




