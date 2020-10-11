#!/usr/bin/env python
# coding: utf-8

# In[444]:


import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# In[445]:



#df    = loader('C:/Users/Emil/Desktop/Parcial/irisdata.txt' )
df = pd.read_table('irisdata.txt', skiprows=9, header = None)
df
df.columns


# In[446]:


X= df.iloc[:,0:4].values
y = df.iloc[:,4].values


X_std = StandardScaler().fit_transform(X)

# In[447]:


z = np.cov(X_std.T)
z


# In[448]:


eig = la.eig(z)
eigval=eig[0]
eigvec=eig[1]
eigval
eig_pairs = [(np.abs(eigval[i]), eigvec[:,i]) for i in range(len(eigval))]

matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

matrix_z = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1),
                      eig_pairs[2][1].reshape(4,1)))

Y = X.dot(matrix_w)
Z = X.dot(matrix_z)

# In[449]:



sume = (sum(eigval)).real
sume


# In[450]:


pc=(eigval/sume).real
por=pc


# In[451]:


py=np.arange(4)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
lang=['PC1','PC2','PC3','PC4']
ax.bar(lang,por)
plt.show()



# In[452]:






# In[454]:


#colum3


# In[455]:

 
with plt.style.context('seaborn-whitegrid'):   
    
    for lab2, col2 in zip((0, 1, 2),
                        ('magenta', 'cyan', 'limegreen')):
            ax  =  plt.axes(projection='3d')
            ax.scatter3D(Z[y==lab2, 0],Z[y==lab2, 1],Z[y==lab2, 2], label=lab2,c=col2 )
            plt.show()




# In[ ]:

with plt.style.context('seaborn-whitegrid'):

    for lab, col in zip((0, 1, 2),
                        ('magenta', 'cyan', 'limegreen')):
        plt.scatter(Y[y==lab, 0],Y[y==lab, 1],label=lab,c=col)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()



# In[ ]:




# In[ ]:




