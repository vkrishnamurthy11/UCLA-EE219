#Author: Akhil Avula
#Problem 3
#LSI
import numpy as np
from scipy.linalg import svd

U, s, VT = svd(X) #X is TF-IDF matrix from problem 2
# s[::-1].sort() # use this method or the one below
-np.sort(-s)

V = VT.transpose()
k = 50
Uk = U[:,:k]
Vk = V[:,:k]

np.matmul(X,Vk)

#NMF
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
from sklearn.decomposition import NMF
model = NMF(n_components=k, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_
# not sure if we need to solve a least squared problem or if the NMF function takes care of it

# Compare the norms of NMF and LSI
