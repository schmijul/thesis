#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from data_preparation import *
from interpolation import *

# Kriging
from pykrige.uk import UniversalKriging
import skgstat as skg

#plotting utils

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


length = 100
sampling_distance_x = 4
sampling_distance_y = 24
verbose = False


# In[3]:


whole_map = pd.read_csv('WholeMap_Rounds_40_to_17.csv')
map = stack_map(whole_map) # create a stacked map dataframe with columns x, y, z
map = cut_map_len(map,length) # cut the map to the length of the map
data_i, data_o = resample(map, sampling_distance_x, sampling_distance_y) # resample the map


# In[17]:


map_whole = map.pivot_table(index='y', columns='x', values='z')


# In[7]:


# PYKrig with UniversalKriging

kriging_type = 'UK'

uk_pykrige,uk_pykrige_stacked = kriging(data_i, data_o, type=kriging_type)

# PYKrig with OrdinaryKriging

kriging_type = 'OK'

ok_pykrige, ok_pykrige_stacked = kriging(data_i, data_o, type=kriging_type)

# SKG Ordinarykriging

V = skg.Variogram(data_i[['x', 'y']], data_i['z'], n_lags=10)
ok_exact = skg.OrdinaryKriging(V, mode='exact')
ok_estimate = skg.OrdinaryKriging(V, mode='estimate', precision=100)

skg_stacked_exact = ok_exact.transform(data_o[['x', 'y']])
skg_stacked_exact = pd.DataFrame([skg_stacked_exact,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
skg_stacked_exact.columns = ['z','x','y']
skg_stacked_exact   = skg_stacked_exact[['x','y','z']]

skg_exact = skg_stacked_exact.pivot_table(index='y', columns='x', values='z')

skg_stacked_estimate = ok_estimate.transform(data_o[['x', 'y']])
skg_stacked_estimate = pd.DataFrame([skg_stacked_estimate,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
skg_stacked_estimate.columns = ['z','x','y']
skg_stacked_estimate   = skg_stacked_estimate[['x','y','z']]

skg_estimate = skg_stacked_estimate.pivot_table(index='y', columns='x', values='z')


# In[15]:




fig, axes = plt.subplots(3,2, figsize=(10, 18))
sns.heatmap(ax=axes[0,0],data=data_o.pivot_table(index='y', columns='x', values='z'))
axes[0,0].title.set_text('ground truth')
sns.heatmap(ax=axes[1,0],data=uk_pykrige)
axes[1,0].title.set_text('py krige UK')
sns.heatmap(ax=axes[1,1],data=ok_pykrige)
axes[1,1].title.set_text('py krige OK')
sns.heatmap(ax=axes[2,0],data=skg_exact)
axes[2,0].title.set_text('skg exact')
sns.heatmap(ax=axes[2,1],data=skg_estimate)
axes[2,1].title.set_text('skg estimate')

plt.savefig('Kriging_comparison.png')

    



# In[23]:


# PYKrig with UniversalKriging

kriging_type = 'UK'

uk_pykrige,uk_pykrige_stacked = kriging(map, data_o, type=kriging_type)

# PYKrig with OrdinaryKriging

kriging_type = 'OK'

ok_pykrige, ok_pykrige_stacked = kriging(map, data_o, type=kriging_type)

# SKG Ordinarykriging

V = skg.Variogram(map[['x','y']], map['z'], n_lags=10)
ok_exact = skg.OrdinaryKriging(V, mode='exact')
ok_estimate = skg.OrdinaryKriging(V, mode='estimate', precision=100)

skg_stacked_exact = ok_exact.transform(data_o[['x', 'y']])
skg_stacked_exact = pd.DataFrame([skg_stacked_exact,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
skg_stacked_exact.columns = ['z','x','y']
skg_stacked_exact   = skg_stacked_exact[['x','y','z']]

skg_exact = skg_stacked_exact.pivot_table(index='y', columns='x', values='z')

skg_stacked_estimate = ok_estimate.transform(data_o[['x', 'y']])
skg_stacked_estimate = pd.DataFrame([skg_stacked_estimate,data_o['x'].to_numpy(),data_o['y'].to_numpy()]).transpose()
skg_stacked_estimate.columns = ['z','x','y']
skg_stacked_estimate   = skg_stacked_estimate[['x','y','z']]

skg_estimate = skg_stacked_estimate.pivot_table(index='y', columns='x', values='z')


# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(20, 14))
sns.heatmap(ax=axes[0,0],data=data_o.pivot_table(index='y', columns='x', values='z'))
axes[0,0].title.set_text('ground truth')
sns.heatmap(ax=axes[1,0],data=uk_pykrige)
axes[1,0].title.set_text('py krige UK')
sns.heatmap(ax=axes[1,1],data=ok_pykrige)
axes[1,1].title.set_text('py krige OK')
sns.heatmap(ax=axes[2,0],data=skg__exact)
axes[2,0].title.set_text('skg exact')
sns.heatmap(ax=axes[2,1],data=skg_estimate)
axes[2,1].title.set_text('skg estimate')

plt.savefig('Kriging_comparison_predict_whole_map.png')




# In[ ]:




