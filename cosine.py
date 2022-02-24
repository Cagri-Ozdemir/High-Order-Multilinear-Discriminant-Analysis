import numpy as np
from numpy import linalg

import scipy
from scipy.fftpack import dct, idct
from scipy.linalg import circulant

def tSVDdct(A):
    (n0,n1,n2) = A.shape
    U1 = np.zeros((n0, n1, n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0, n2, n2))
    A = dct(A, type=1,axis=0, norm='ortho')

    for i in range(n0):
      (U1[i,:,:], S, V1[i,:,:]) = np.linalg.svd(A[i,:,:],full_matrices='true')
      np.fill_diagonal(S1[i,:,:], S)
      V1[i,:,:] = np.transpose(V1[i,:,:])

    U1 = idct(U1, type=1,axis=0, norm='ortho')
    S1 = idct(S1, type=1,axis=0, norm='ortho')
    V1 = idct(V1, type=1,axis=0, norm='ortho')

    return U1, S1, V1

def tproddct(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    C = np.zeros((n0,n1,nc))
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    D = scipy.fftpack.dct(A, type=1, axis=0,norm='ortho')
    Bhat = scipy.fftpack.dct(B, type=1, axis=0,norm='ortho')

    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])

    C1 = scipy.fftpack.idct(C,type=1, axis=0,norm='ortho')

    return C1



def ttransx(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b))

    for j in range(a):
        B[j,:,:] = np.transpose(A[j,:,:])
    return B

def tinvdct(A):
    (n0, n1, n2) = A.shape
    D = scipy.fftpack.dct(A, type=1, axis=0,norm='ortho')

    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])

    D3 = scipy.fftpack.idct(D2, type=1, axis=0, norm='ortho')
    return D3

