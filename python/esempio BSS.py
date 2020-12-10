#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sweep_poly, chirp, spectrogram

from sklearn.decomposition import FastICA, PCA

# Generate the data
np.random.seed(1)
n = 1000
t = np.linspace(0, 10, n)
p = np.poly1d([0.025, -0.36, 1.25, 2.0])

s1 = np.cos(2 * t)  # Signal 1 
s2 = sweep_poly(t, p) # Signal 2 
s3 = chirp(t, f0=6, f1=1, t1=10, method='linear') # Signal 3

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardized data

# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Observations

# ICA
ica = FastICA(n_components=3)
S_ = ica.fit_transform(X)  
A_ = ica.mixing_  # estimated mixing matrix

# PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)

#Plot
plt.figure()

models = [S, X, S_, H]
names = ['True signals',
         'Observations',
         'ICA signals',
         'PCA signals']
colors = ['red', 'orange', 'steelblue']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()


# In[ ]:




