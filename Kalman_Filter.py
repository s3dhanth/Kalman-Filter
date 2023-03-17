#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


# In[14]:


data2 = pd.read_csv('SBIN1.csv')


# In[22]:


sbi = data2.Close


# In[35]:


sbi


# In[4]:


data =pd.read_csv('ICICIBANK1.csv')


# In[23]:


icici = data


# In[28]:


icici.corr(sbi)


# In[29]:


sbi.corr(icici)


# In[33]:


dataframe = pd.concat([icici,sbi],axis = 1)


# In[41]:


dataframe.columns = ['icici','sbi']


# In[48]:


dataframe['icici']


# In[49]:


plt.scatter(dataframe.icici,dataframe.sbi)


# In[ ]:


delta = 1e-5
trans_cov = delta / (1 - delta) * np.eye(2)
obs_mat = np.vstack([dataframe.sbi], np.ones(dataframe.sbi.shape)).T[:, np.newaxis]
    
kf = KalmanFilter(
        n_dim_obs=1, 
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov)
    
state_means, state_covs = kf.filter(dataframe['icici'].values)


# In[ ]:


pd.DataFrame(
        dict(
            slope=state_means[:, 0], 
            intercept=state_means[:, 1]
        ), index=prices.index
    ).plot(subplots=True)
    plt.show()
    

