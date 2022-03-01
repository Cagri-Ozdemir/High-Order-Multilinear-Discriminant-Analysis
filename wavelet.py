import numpy as np
from numpy import linalg
import pywt
from sklearn.decomposition import  FastICA,fastica

def tSVDdwt(A):
    (n0,n1,n2) = A.shape
    z1 =int(n0)
    z2 = int(n0/2)
    U1 = np.zeros((n0,n1,n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0,n2,n2))
    coeffs = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffs
    #cA = np.zeros((z2,n1,n2))
    arr = np.concatenate((cA, cD),axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      U, S, Vt = np.linalg.svd(M,full_matrices=True)
      np.fill_diagonal(S1[i,:,:],S)
      V1[i, :, :] = Vt.T
      U1[i,:,:] = U
    cU1 = U1[0:z2, :, :]; cU2 = U1[z2:z1, :, :]
    cS1 = S1[0:z2, :, :]; cS2 = S1[z2:z1, :, :]
    cV1 = V1[0:z2, :, :]; cV2 = V1[z2:z1, :, :]
    U1 = pywt.idwt(cU1,cU2, 'haar', axis=0)
    S1 = pywt.idwt(cS1,cS2, 'haar', axis=0)
    V1 = pywt.idwt(cV1,cV2, 'haar', axis=0)
    return U1, S1, V1

def tproddwt(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    C = np.zeros((n0, n1, nc))
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    #cA = np.zeros((z2, n1, n2))
    D = np.concatenate((cA, cD),axis=0)
    coeffsB = pywt.dwt(B, 'haar', axis=0)
    cAA, cDD = coeffsB
    #cAA = np.zeros((z2, nb, nc))
    Bhat = np.concatenate((cAA, cDD),axis=0)
    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])
    cC1 = C[0:z2, :, :]; cC2 = C[z2:z1, :, :]
    Cx = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return Cx
def tinvdwt(A):
    (n0, n1, n2) = A.shape
    z1 = int(n0)
    z2 = int(n0 / 2)
    coeffsA = pywt.dwt(A, 'haar', axis=0)
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z2, :, :]
    cC2 = D2[z2:z1, :, :]

    D3 = pywt.idwt(cC1, cC2, 'haar', axis=0)
    return D3

def tSVDdwt2(A):
    (n0,n1,n2) = A.shape
    z3 =int(n0)
    z2 = int(n0/2)
    z1 = int(n0/4)
    U1 = np.zeros((n0,n1,n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0,n2,n2))
    coeffs = pywt.wavedec(A, 'haar', axis=0,level=2)
    arr = np.concatenate((coeffs[0],coeffs[1],coeffs[2]),axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      U, S, Vt = np.linalg.svd(M,full_matrices=True)
      np.fill_diagonal(S1[i,:,:],S)
      V1[i, :, :] = Vt.T
      U1[i,:,:] = U
    cU1 = U1[0:z1, :, :]; cU2 = U1[z1:z2, :, :];cU3 = U1[z2:z3, :, :]
    cU = [cU1, cU2, cU3]
    cS1 = S1[0:z1, :, :]; cS2 = S1[z1:z2, :, :];cS3 = S1[z2:z3, :, :]
    cS = [cS1, cS2, cS3]
    cV1 = V1[0:z1, :, :]; cV2 = V1[z1:z2, :, :];cV3 = V1[z2:z3, :, :]
    cV = [cV1, cV2, cV3]
    U1 = pywt.waverec(cU, 'haar', axis=0)
    S1 = pywt.waverec(cS, 'haar', axis=0)
    V1 = pywt.waverec(cV, 'haar', axis=0)
    return U1, S1, V1
def tproddwt2(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    z3 = int(n0)
    z2 = int(n0 / 2)
    z1 = int(n0 / 4)
    C = np.zeros((n0, n1, nc))
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.wavedec(A, 'haar', axis=0,level=2)
    D = np.concatenate((coeffsA[0],coeffsA[1],coeffsA[2]),axis=0)
    coeffsB = pywt.wavedec(B, 'haar', axis=0,level=2)
    Bhat = np.concatenate((coeffsB[0], coeffsB[1],coeffsB[2]),axis=0)
    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])
    cC1 = C[0:z1, :, :]; cC2 = C[z1:z2, :, :];cC3=C[z2:z3,:,:]
    cC = [cC1,cC2,cC3]
    Cx = pywt.waverec(cC, 'haar', axis=0)
    return Cx
def tinvdwt2(A):
    (n0, n1, n2) = A.shape
    z3 = int(n0)
    z2 = int(n0 / 2)
    z1 = int(n0 / 4)
    coeffsA = pywt.wavedec(A, 'haar', axis=0,level=2)
    D = np.concatenate((coeffsA[0],coeffsA[1],coeffsA[2]),axis=0)
    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z1, :, :];  cC2 = D2[z1:z2, :, :];  cC3 = D2[z2:z3, :, :]
    cC = [cC1, cC2, cC3]
    D3 = pywt.waverec(cC, 'haar', axis=0)
    return D3
def tSVDdwt3(A):
    (n0,n1,n2) = A.shape
    z4= int(n0/2)
    z3 = int(n0/4)
    z2 = int(n0/8)
    z1 = int(n0/8)
    U1 = np.zeros((n0,n1,n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0,n2,n2))
    coeffs = pywt.wavedec(A, 'haar', axis=0,level=3)
    arr = np.concatenate((coeffs[0],coeffs[1],coeffs[2],coeffs[3]),axis=0)
    for i in range(n0):
      M = arr[i, 0:]
      U, S, Vt = np.linalg.svd(M,full_matrices=True)
      np.fill_diagonal(S1[i,:,:],S)
      V1[i, :, :] = Vt.T
      U1[i,:,:] = U
    cU1 = U1[0:z1, :, :]; cU2 = U1[z1:z2+z1, :, :];cU3 = U1[z2+z1:z4, :, :];cU4 = U1[z4:n0, :, :]
    cU = [cU1, cU2, cU3,cU4]
    cS1 = S1[0:z1, :, :]; cS2 = S1[z1:z2+z1, :, :];cS3 = S1[z2+z1:z4, :, :];cS4 = S1[z4:n0, :, :]
    cS = [cS1, cS2, cS3,cS4]
    cV1 = V1[0:z1, :, :]; cV2 = V1[z1:z2+z1, :, :];cV3 = V1[z2+z1:z4, :, :];cV4 = V1[z4:n0, :, :]
    cV = [cV1, cV2, cV3,cV4]
    U1 = pywt.waverec(cU, 'haar', axis=0)
    S1 = pywt.waverec(cS, 'haar', axis=0)
    V1 = pywt.waverec(cV, 'haar', axis=0)
    return U1, S1, V1

def tproddwt3(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    z4 = int(n0 / 2)
    z3 = int(n0 / 4)
    z2 = int(n0 / 8)
    z1 = int(n0 / 8)
    C = np.zeros((n0, n1, nc))
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.wavedec(A, 'haar', axis=0,level=3)
    D = np.concatenate((coeffsA[0],coeffsA[1],coeffsA[2],coeffsA[3]),axis=0)
    coeffsB = pywt.wavedec(B, 'haar', axis=0,level=3)
    Bhat = np.concatenate((coeffsB[0], coeffsB[1],coeffsB[2],coeffsB[3]),axis=0)
    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])
    cC1 = C[0:z1, :, :]; cC2 = C[z1:z2+z1, :, :];cC3=C[z3:z4,:,:];cC4=C[z4:n0,:,:]
    cC = [cC1,cC2,cC3,cC4]
    Cx = pywt.waverec(cC, 'haar', axis=0)
    return Cx
def tinvdwt3(A):
    (n0, n1, n2) = A.shape
    z4 = int(n0 / 2)
    z3 = int(n0 / 4)
    z2 = int(n0 / 8)
    z1 = int(n0 / 8)
    coeffsA = pywt.wavedec(A, 'haar', axis=0,level=3)
    D = np.concatenate((coeffsA[0],coeffsA[1],coeffsA[2],coeffsA[3]),axis=0)
    D2 = np.zeros((n0,n1,n2))
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z1, :, :]; cC2 = D2[z1:z2 + z1, :, :];  cC3 = D2[z3:z4, :, :];  cC4 = D2[z4:n0, :, :]
    cC = [cC1, cC2, cC3, cC4]
    D3 = pywt.waverec(cC, 'haar', axis=0)
    return D3

def ttransx(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b))

    for j in range(a):
        B[j,:,:] = np.transpose(A[j,:,:])
    return B

def tSVDdwtdb_4(A):
    coeffs = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    arr = np.concatenate((coeffs[0], coeffs[1]), axis=0)
    (n0,n1,n2) =arr.shape
    U1 = np.zeros((n0, n1, n1))
    S1 = np.zeros((n0, n1, n2))
    V1 = np.zeros((n0, n2, n2))
    for i in range(n0):
        M = arr[i, 0:]
        U, S, Vt = np.linalg.svd(M, full_matrices=True)
        np.fill_diagonal(S1[i, :, :], S)
        V1[i, :, :] = Vt.T
        U1[i, :, :] = U
    z2 = int(n0/2)
    cU1 = U1[0:z2, :, :];    cU2 = U1[z2:n0, :, :]
    cS1 = S1[0:z2, :, :];    cS2 = S1[z2:n0, :, :]
    cV1 = V1[0:z2, :, :];    cV2 = V1[z2:n0, :, :]
    U1 = pywt.idwt(cU1, cU2, 'db4', axis=0,mode='periodization')
    S1 = pywt.idwt(cS1, cS2, 'db4', axis=0,mode='periodization')
    V1 = pywt.idwt(cV1, cV2, 'db4', axis=0,mode='periodization')
    return U1, S1, V1


def tproddwtdb4(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    coeffsA = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    D = np.concatenate((coeffsA[0],coeffsA[1]), axis=0)
    coeffsB = pywt.dwt(B, 'db4', axis=0,mode='periodization')
    Bhat = np.concatenate((coeffsB[0], coeffsB[1]), axis=0)
    (z1,z2,z3) = Bhat.shape
    C = np.zeros((z1,n1,nc))
    for i in range(z1):
        C[i, :, :] = np.matmul(D[i, :, :], Bhat[i, :, :])
    cC1 = C[0:int(z1/2), :, :];    cC2 = C[int(z1/2):z1, :, :]
    Cx = pywt.idwt(cC1, cC2, 'db4', axis=0,mode='periodization')
    return Cx
def tinvdwt_db4(A):
    (n0, n1, n2) = A.shape
    z1 =int((n0+6)/2)
    coeffsA = pywt.dwt(A, 'db4', axis=0,mode='periodization')
    cA, cD = coeffsA
    D = np.concatenate((cA, cD))
    D2 = np.zeros((z1+z1,n1,n2))
    for i in range(z1+z1):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])
    cC1 = D2[0:z1, :, :]
    cC2 = D2[z1:z1+z1, :, :]

    D3 = pywt.idwt(cC1, cC2, 'db4', axis=0,mode='periodization')
    return D3



