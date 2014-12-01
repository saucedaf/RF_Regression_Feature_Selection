
# coding: utf-8

# In[3]:

import numpy as np
import sklearn as sk
#import sklearn.datasets as skd
import sklearn.ensemble as ske
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().magic(u'matplotlib inline')
mpl.rcParams['figure.dpi'] = mpl.rcParams['savefig.dpi'] = 300


# In[4]:

import pandas as pd
import numpy as np
from __future__ import division
from sklearn.preprocessing import StandardScaler
churn_df = pd.read_csv('path')

churn_df


# In[5]:

churn_result = churn_df['Canceled']
y = np.where(churn_result == 'Canceled',1,0)

y


# In[6]:

to_drop = ['items to drop']
X = churn_df.drop(to_drop,axis=1)
X_droped = X.as_matrix().astype(np.float)
X_droped


# In[7]:

col_names = X.columns.tolist()

col_names

col_names_array = np.asarray(col_names)

col_names_array


# In[8]:

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_droped)

X_normalized


# In[9]:

y


# In[10]:

data = {'data':X_normalized, 'feature_names':col_names_array, 'target':y}
X = data['data']
y = data['target']
reg = ske.RandomForestRegressor()
reg.fit(X, y);


# In[13]:

fet_ind = np.argsort(reg.feature_importances_)[::-1]
fet_imp = reg.feature_importances_[fet_ind]
fet_imp


# In[48]:

fig = plt.figure(figsize=(30,20));
ax = plt.subplot(111);
plt.bar(np.arange(len(fet_imp)), fet_imp, width=1, lw=2);
plt.grid(False);

ax.set_xticks(np.arange(len(fet_imp))+.5);
ax.set_xticklabels(data['feature_names'][fet_ind]);

plt.xlim(0, len(fet_imp));


# In[ ]:



