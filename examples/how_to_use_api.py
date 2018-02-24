
# coding: utf-8

# In[1]:


# load packages
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import ineqpy as inq

get_ipython().run_line_magic('matplotlib', 'inline')


# # First-steps

# In[2]:


# load data
data = pd.read_csv('eusilc.csv', index_col=0).dropna()
svy = inq.api.Survey(data, weights='rb050')


# In[3]:


svy.gini('eqincome')


# In[4]:


svy.atkinson('eqincome')


# In[5]:


svy.theil('eqincome')


# In[6]:


svy.mean('eqincome')


# In[7]:


svy.percentile('eqincome')


# In[8]:


svy.kurt('eqincome')


# In[9]:


svy.skew('eqincome')


# In[10]:


svy.lorenz('eqincome').plot(figsize=(5,5))
# plt.show()
