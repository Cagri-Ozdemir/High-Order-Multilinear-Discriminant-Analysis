import numpy as np
from numpy import linalg

def ttrans(A):
    (a, b, c) = A.shape
    B = np.zeros((a,c,b))
    B[0,:,:] = np.transpose(A[0,:,:])
    for j in range(a-1,0,-1):
        B[a-1-j+1,:,:] = np.transpose(A[j,:,:])
    return B


def tSVD(A):
    n0,n1,n2 = A.shape
    U1 = np.zeros((n0,n1,n1),dtype=complex)
    S1 = np.zeros((n0, n1, n2),dtype=complex)
    V1 = np.zeros((n0,n2,n2),dtype=complex)
    A = np.fft.fft(A, axis=0)

    for i in range(n0):
      (U, S, Vt) = np.linalg.svd(A[i,:,:],full_matrices='true')
      np.fill_diagonal(S1[i,:,:],S)
      U1[i,:,:] = U
      Vc = np.conj(Vt)
      V1[i,:,:] = Vc.T

    U1x = np.real(np.fft.ifft(U1, axis=0))
    S1x = np.real(np.fft.ifft(S1, axis=0))
    V1x = np.real(np.fft.ifft(V1, axis=0))


    return U1x, S1x, V1x



def tprod(A,B):
    (n0, n1, n2) = A.shape
    (na, nb, nc) = B.shape
    C = np.zeros((n0,n1,nc), dtype=complex)
    if n0 != na and n2 != nb:
        print('warning, dimensions are not acceptable')
        return
    D = np.fft.fft(A, axis = 0)
    Bhat = np.fft.fft(B, axis = 0)

    for i in range(n0):
       C[i,:,:] = np.matmul(D[i,:,:],Bhat[i,:,:])

    C = np.fft.ifft(C, axis=0)
    Cx = np.real(C)

    return Cx

def tinvx(A):
    (n0, n1, n2) = A.shape
    D = np.fft.fft(A, axis=0)

    D2 = np.zeros((n0,n1,n2),dtype='complex')
    for i in range(n0):
        D2[i,:,:]=np.linalg.inv(D[i,:,:])

    D3 = np.real(np.fft.ifft(D2, axis=0))
    return D3


def fronorm(A):
    tmp = A*A
    B = np.absolute(tmp)
    C = np.sum(B)
    y = np.sqrt(C)
    return y

#A = np.random.rand(10,10,10)
#u,s,v = tSVD(A)
#orth1 = u[0,:,:].T@u[0,:,:]

#uu = np.fft.fft(u,axis=0).real
#orth2 = uu[0,:,:].T@uu[0,:,:]