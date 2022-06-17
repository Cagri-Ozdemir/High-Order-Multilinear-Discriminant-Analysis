import numpy as np
import pywt
from scipy import linalg

def teigdwt4(A):
    (n0,n1,n2,n3) = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    U1 = np.zeros((n0,n1,n2,n2))
    S1 = np.zeros((n0, n1, n2, n3))

    coeffs = pywt.dwt(A, 'haar', axis=1)
    cA, cD = coeffs
    arr = np.concatenate((cA, cD),axis=1)
    coeffs2 = pywt.dwt(arr, 'haar', axis=0)
    cA2, cD2 = coeffs2
    arr2 = np.concatenate((cA2, cD2),axis=0)
    for i in range(n0):
        for j in range(n1):
            s, u = np.linalg.eig(arr2[i, j, :, :])
            idx = np.argsort(s)
            idx = idx[::-1][:n2]
            s = s[idx]
            u = u[:, idx]
            s, u = linalg.cdf2rdf(s, u)
            S1[i, j, :, :] = s
            U1[i, j, :, :] = u
    cU1 = U1[0:z2, :, :,:]; cU2 = U1[z2:z1, :, :,:]
    cS1 = S1[0:z2, :, :,:]; cS2 = S1[z2:z1, :, :,:]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    cU1 = U1[:, 0:z22, :, :];
    cU2 = U1[:, z22:z11, :, :]
    cS1 = S1[:, 0:z22, :, :];
    cS2 = S1[:, z22:z11, :, :]
    U1 = pywt.idwt(cU1, cU2, 'haar', axis=1)
    S1 = pywt.idwt(cS1, cS2, 'haar', axis=1)
    return S1,U1

def tproddwt4(A,B):
    (n0, n1, n2, n3) = A.shape
    (na, nb, nc, nd) = B.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    C = np.zeros((n0, n1, n2, nd))
    if n0 != na and n3 != nc:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'haar', axis=1)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD),axis=1)
    coeffsB = pywt.dwt(B, 'haar', axis=1)
    cAA, cDD = coeffsB
    Bhat = np.concatenate((cAA, cDD),axis=1)

    coeffsA = pywt.dwt(D, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD), axis=0)
    coeffsB = pywt.dwt(Bhat, 'haar', axis=0)
    cAA, cDD = coeffsB
    Bhat = np.concatenate((cAA, cDD), axis=0)
    for i in range(n0):
        for j in range(n1):
            C[i, j, :, :] = D[i, j, :, :] @ Bhat[i, j, :, :]
    cC1 = C[0:z2, :, :,:]; cC2 = C[z2:z1, :, :,:]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)

    cC1 = Cx[:, 0:z22, :,:];
    cC2 = Cx[:, z22:z11, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=1)
    return Cx


def ttransdwt4(A):
    n0, n1, n2, n3 = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    coeffs = pywt.dwt(A, 'haar', axis=1)
    cA, cD = coeffs
    arr = np.concatenate((cA, cD), axis=1)
    coeffs2 = pywt.dwt(arr, 'haar', axis=0)
    cA2, cD2 = coeffs2
    arr2 = np.concatenate((cA2, cD2), axis=0)
    B = np.zeros((n0,n1,n3,n2))
    for i in range(n0):
        for j in range(n1):
            B[i,j,:,:] = np.transpose((arr2[i,j,:,:]))
    cC1 = B[0:z2, :, :, :];
    cC2 = B[z2:z1, :, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)

    cC1 = Cx[:, 0:z22, :, :];
    cC2 = Cx[:, z22:z11, :, :]
    B2 = pywt.idwt(cC1, cC2, 'haar', axis=1)
    return B2
def tinvdwt4(A):
    n0, n1, n2, n3 = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    coeffsA = pywt.dwt(A, 'haar', axis=1)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD),axis=1)

    coeffsA = pywt.dwt(D, 'haar', axis=0)
    cA, cD = coeffsA
    DD = np.concatenate((cA, cD), axis=0)

    D2 = np.zeros((n0,n1,n2,n3))
    for i in range(n0):
        for j in range(n1):
            D2[i, j, :, :] = np.linalg.inv(DD[i, j, :, :])
    cC1 = D2[0:z2, :, :,:]
    cC2 = D2[z2:z1, :, :,:]
    D3 = pywt.idwt(cC1, cC2, 'haar', axis=0)

    cC1 = D3[:, 0:z22, :,:]
    cC2 = D3[:, z22:z11, :,:]
    D4 = pywt.idwt(cC1, cC2, 'haar', axis=1)
    return D4
def Class_scatters_dwtcomp4(num_class,Tensor_train,y_train):
    n0,n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n0,n1,n2,num_class))
    Sw =  np.zeros((n0,n1,n2,n2))
    Sb = np.zeros((n0,n1, n2, n2))
    a = np.zeros((n0,n1,n2,1))
    b = np.zeros((n0,n1, n2, 1))
    Mean_tensor = np.zeros((n0,n1, n2, 1))
    Mean_tensor[:,:,:,0] = (Tensor_train.sum(axis=3))/n3
    for i in range(num_class):
      Sa = np.zeros((n0,n1, n2, n2))
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,:,i] = (Tensor_train[:,:,:,idx].sum(axis=3))/occurrences
      for j in idx:
          a[:,:,:,0] = Tensor_train[:,:,:,j]-mean_tensor_train[:,:,:,i]
          Sa = Sa + tproddwt4(a,ttransdwt4(a))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,:,0] = mean_tensor_train[:,:,:,i] - Mean_tensor[:,:,:,0]
        Sb = Sb + (tproddwt4(b,ttransdwt4(b)))*occurrences

    return Sw,Sb

def updated_Sw_dwt(Sw):
    n0, n1, n2, n2 = Sw.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    coeffsA = pywt.dwt(Sw, 'haar', axis=1)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD), axis=1)

    coeffsA = pywt.dwt(D, 'haar', axis=0)
    cA, cD = coeffsA
    DD = np.concatenate((cA, cD), axis=0)
    U = np.zeros((n0, n1, n2, n2))
    S = np.zeros((n0, n1, n2, n2))
    k=0
    for i in range(n0):
        for j in range(n1):
            cond = np.linalg.cond(DD[i, j, :, :])
            if 1 / cond <= 1.e-5:
                k += 1
                s, u = np.linalg.eig(DD[i, j, :, :])
                idx = np.argsort(s)
                idx = idx[::-1][:n2]
                s = s[idx]
                u = u[:, idx]
                Em = 0
                M = 0
                while Em <= 0.98:
                    Em = np.sum(s[:M]) / np.sum(s)
                    M += 1
                lamda_star = (1 / (n2 - M)) * np.sum(s[M + 1:])
                s[M + 1:] = lamda_star
                np.fill_diagonal(S[i, j, :, :], s)
                U[i, j, :, :] = u
            else:
                s, u = np.linalg.eig(DD[i, j, :, :])
                idx = np.argsort(s)
                idx = idx[::-1][:n2]
                s = s[idx]
                u = u[:, idx]
                np.fill_diagonal(S[i, j, :, :], s)
                U[i, j, :, :] = u
    cC1 = U[0:z2, :, :, :]
    cC2 = U[z2:z1, :, :, :]
    U2 = pywt.idwt(cC1, cC2, 'haar', axis=0)
    CC1 = U2[:, 0:z22, :, :]
    CC2 = U2[:, z22:z11, :, :]
    U3 = pywt.idwt(CC1, CC2, 'haar', axis=1)

    cC1 = S[0:z2, :, :, :]
    cC2 = S[z2:z1, :, :, :]
    S2 = pywt.idwt(cC1, cC2, 'haar', axis=0)
    CC1 = S2[:, 0:z22, :, :]
    CC2 = S2[:, z22:z11, :, :]
    S3 = pywt.idwt(CC1, CC2, 'haar', axis=1)

    ww1 = tproddwt4(U3, S3)
    Sww1 = tproddwt4(ww1, ttransdwt4(U3))
    return Sww1,k

def condition_dwt(Sw):
    n0, n1, n2, n2 = Sw.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    z11 = int(n1)
    z22 = int(n1 / 2)
    coeffsA = pywt.dwt(Sw, 'haar', axis=1)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD), axis=1)

    coeffsA = pywt.dwt(D, 'haar', axis=0)
    cA, cD = coeffsA
    DD = np.concatenate((cA, cD), axis=0)
    k=0
    condition = np.zeros((n0*n1))
    for i in range(n0):
        for j in range(n1):
            cond = np.linalg.cond(DD[i, j, :, :])
            condition[k] = cond
            k+=1
    return condition