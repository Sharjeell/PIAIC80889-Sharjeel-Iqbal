#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
print(np.__version__)


# In[5]:


arr=np.arange(10)
arr


# In[7]:


np.full((3,3),True,dtype=bool)


# In[8]:


# alternative method
np.ones((3,3), dtype=bool)


# In[10]:


arr= np.array([0,1,2,3,4,5,6,7,8,9])
arr[arr % 2==1]


# In[15]:


arr=np.array([0,1,2,3,4,5,6,7,8,9])
arr[arr % 2 == 1] = - 1
arr


# In[17]:


arr=np.arange(10)
out=np.where(arr % 2 == 1, -1, arr)
print(arr)
out


# In[18]:


arr=np.arange(10)
arr.reshape(2,-1)


# In[22]:


a=np.arange(10).reshape(2,-1)
b=np.repeat(1,10).reshape(2,-1)
np.vstack([a,b]) #vertical stack


# In[23]:


a=np.arange(10).reshape(2,-1)
b=np.repeat(1,10).reshape(2,-1)

np.hstack([a,b]) #horizontal stack


# In[30]:


a=np.array([1,2,3])
np.r_[np.repeat(a,3),np.title(a,3)]


# In[31]:


a= np.array([1,2,3,2,3,4,3,4,5,6])
b= np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# In[32]:


a=np.array([1,2,3,4,5])
b=np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# In[34]:


a=np.array([1,2,3,2,3,4,3,4,5,6])
b=np.array([7,2,10,2,7,4,9,4,9,4,9,8])
np.where(a == b)


# In[37]:


a=np.arange(15)
index=np.where((a>=5) & (a<=10))
a[index]


# In[6]:


def maxx(x, y):
    if x >= y:
        return x
    else:
        return y

pair_max = np.vectorize(maxx, otypes=[float])

a = np.array([5, 7, 9, 8, 6, 4, 5])
b = np.array([6, 3, 4, 8, 9, 7, 1])

pair_max(a, b)


# In[7]:


arr = np.arange(9).reshape(3,3)
arr

arr[:, [1,0,2]]


# In[8]:


arr = np.arange(9).reshape(3,3)
arr[[1,0,2], :]


# In[9]:


arr = np.arange(9).reshape(3,3)
arr[:, ::-1]


# In[10]:


rand_arr = np.random.random((5,3))
rand_arr = np.random.random([5,3])
np.set_printoptions(precision=3)
rand_arr[:4]


# In[11]:


np.set_printoptions(suppress=False)
np.random.seed(100)
rand_arr = np.random.random([3,3])/1e3
rand_arr


# In[12]:


np.set_printoptions(threshold=6)
a = np.arange(15)
a


# In[14]:



np.set_printoptions(threshold=6)
a = np.arange(15)
np.set_printoptions(threshold=np.nan)
a


# In[15]:



url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='object')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')
iris[:3]


# In[ ]:




