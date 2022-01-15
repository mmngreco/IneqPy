# load packages
import pandas as pd
import numpy as np
import ineqpy as ineq
from pathlib import Path
import matplotlib.pyplot as plt

# inputs
data_path = Path("ineq.__file__").parent / "examples/eusilc.csv"
data = pd.read_csv(data_path, index_col=0).dropna()
svy = ineq.api.Survey(data, weights="rb050")

# In[3]:
colname = "eqincome"
svy.gini(colname)

# In[4]:
svy.atkinson(colname)

# In[5]:
svy.theil(colname)

# In[6]:
svy.mean(colname)

# In[7]:
svy.percentile(colname)

# In[8]:
svy.kurt(colname)

# In[9]:
svy.skew(colname)

# In[10]:
svy.lorenz(colname).plot(figsize=(5, 5))

# In[10]:
# also works passing variables.
x = data.eqincome
w = data.rb050
ineq.var(variable=x, weights=w)
