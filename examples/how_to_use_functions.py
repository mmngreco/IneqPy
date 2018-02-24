# coding: utf-8

# In[1]:

# load packages
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import ineqpy as ineq

get_ipython().run_line_magic('matplotlib', 'inline')


# # First-steps

# In[2]:

# load data

data = pd.read_csv('eusilc.csv', index_col=0).dropna()
x = 'eqincome'
w = 'rb050'
data[[x, w]].head()

# Out[2]:

       eqincome       rb050
1  16090.694444  504.569620
2  16090.694444  504.569620
4  27076.242857  493.382353
5  27076.242857  493.382353
8  19659.530000  868.220418

# In[3]:


ineq.gini(x, w, data)

# Out[3]:

0.2651613316550714

# In[4]:


ineq.atkinson(x, w, data)


# In[5]:


ineq.theil(x, w, data)


# In[6]:


ineq.mean(x, w, data)


# In[7]:


ineq.percentile(x, w, data)


# In[8]:


ineq.kurt(x, w, data)


# In[9]:


ineq.skew(x, w, data)


# In[10]:


ineq.lorenz(x, w, data).plot(figsize=(5,5))
# plt.show()
