# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#integrantes: GABRIEL GOMEZ, EDUARDO DE LA HOZ, STEPHANIA DE LA HOZ


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
import sys
import mpl_toolkits.mplot3d as mpl
from numpy import genfromtxt
from collections import Counter


# %%
def cleaner(df,k):
    return np.array(df[(abs(df[0]-np.mean(df[0])) <= k*np.std(df[0])) & (abs(df[1]-np.mean(df[1])) <= k*np.std(df[1]))])

def gerOutliners(df,k):
    return np.array(df[(abs(df[0]-np.mean(df[0])) > k*np.std(df[0])) | (abs(df[1]-np.mean(df[1])) > k*np.std(df[1]))])
def proy(rData,base):
    u1p=np.dot(rData,base[0])
    u2p=np.dot(rData,base[1])
    u3p=np.dot(rData,base[2])
    u1u1=np.linalg.norm(base[0])
    u2u2=np.linalg.norm(base[1])
    u3u3=np.linalg.norm(base[2])
    u1pdu1=u1p/u1u1
    u2pdu2=u2p/u2u2
    u3pdu3=u3p/u3u3
    sum1=np.zeros((len(rData),3))
    sum2=np.zeros((len(rData),3))
    sum3=np.zeros((len(rData),3))
    for i in range(len(rData)):
        sum1[i]=u1pdu1[i]*base[0,:]
        sum2[i]=u2pdu2[i]*base[1,:]
        sum3[i]=u3pdu3[i]*base[2,:]
    rest=sum1+sum2+sum3
    return rest


# %%
df=pd.read_table('irisdata.txt',skiprows=9,header=None)
df=df.drop(columns=4)
rawdata=np.array(df)
covRawData = np.cov(rawdata.T)
resultRaw = la.eig(covRawData)
rawdata[0,:]


# %%
covRawData


# %%
resultRaw[0].real


# %%
resultRaw[1]


# %%
test=np.array(resultRaw[1])
test[0]


# %%
eugenValue=np.array(resultRaw[0].real)
sumEugen=np.sum(eugenValue)
porEugen=eugenValue/sumEugen
porEugen=porEugen*100


# %%
x=np.arange(4)
fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
lang=['PC1','PC2','PC3','PC4']
ax.bar(lang,porEugen)
plt.show()


# %%
rawtest=df.drop(columns=3)
tt=np.array(test[:,0:3])
t=proy(rawtest,tt)
t


# %%
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter3D(t[:,0], t[:,1], t[:,2],c=t[:,2],cmap='Dark2')
plt.show()


