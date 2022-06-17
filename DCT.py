import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
from scipy.fftpack import dct, idct
def tsvddct4(A):
    n0,n1,n2,n3 = A.shape
    Ahat = dct(A,axis=1,norm='ortho')
    Ahat = dct(Ahat,axis=0,norm='ortho')
    U = np.zeros((n0,n1,n2,n2))
    S = np.zeros((n0,n1,n2,n3))
    V = np.zeros((n0,n1,n3,n3))
    for i in range(n0):
        for j in range(n1):
            u,s,v = np.linalg.svd(Ahat[i,j,:,:],full_matrices=True)
            np.fill_diagonal(S[i,j, :, :], s)
            U[i,j,:,:] = u
            V[i,j,:,:] = (np.conj(v)).T
    U1 = idct(U,axis=0,norm='ortho')
    U2 = idct(U1,axis=1,norm='ortho')
    S1 = idct(S, axis=0,norm='ortho')
    S2 = idct(S1, axis=1,norm='ortho')
    V1 = idct(V, axis=0,norm='ortho')
    V2 = idct(V1, axis=1,norm='ortho')
    return U2,S2,V2
def teigdct4(A):
    n0, n1, n2, n3 = A.shape
    Ahat = dct(A, axis=1,norm='ortho')
    Ahat = dct(Ahat, axis=0,norm='ortho')
    U = np.zeros((n0, n1, n2, n2))
    S = np.zeros((n0, n1, n2, n3))
    for i in range(n0):
        for j in range(n1):
            s,u = np.linalg.eig(Ahat[i,j,:,:])
            idx = np.argsort(s)
            idx = idx[::-1][:n2]
            s = s[idx]
            u = u[:, idx]
            s, u = linalg.cdf2rdf(s, u)
            #np.fill_diagonal(S[i,j, :, :], s)
            S[i, j, :, :] = s
            U[i,j,:,:] = u
    U1 = idct(U, axis=0,norm='ortho')
    U2 = idct(U1, axis=1,norm='ortho')
    S1 = idct(S, axis=0,norm='ortho')
    S2 = idct(S1, axis=1,norm='ortho')
    return S2,U2
###################
def updated_Sw(Sw):
    n0, n1, n2, n2 = Sw.shape
    Shat = dct(Sw, axis=1,norm='ortho')
    Shat = dct(Shat, axis=0,norm='ortho')
    U = np.zeros((n0, n1, n2, n2))
    S = np.zeros((n0, n1, n2, n2))
    k=0
    for i in range(n0):
        for j in range(n1):
            cond = np.linalg.cond(Shat[i, j, :, :])
            if 1 / cond <= 1.e-5:
                k += 1
                s, u = np.linalg.eig(Shat[i, j, :, :])
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
                s, u = np.linalg.eig(Shat[i, j, :, :])
                idx = np.argsort(s)
                idx = idx[::-1][:n2]
                s = s[idx]
                u = u[:, idx]
                np.fill_diagonal(S[i, j, :, :], s)
                U[i, j, :, :] = u
    U1 = idct(U, axis=0,norm='ortho')
    U2 = idct(U1, axis=1,norm='ortho')
    S1 = idct(S, axis=0,norm='ortho')
    S2 = idct(S1, axis=1,norm='ortho')
    ww1 = tproddct4(U2, S2)
    Sww1 = tproddct4(ww1, ttransdct4(U2))
    return Sww1,k
##################
def condition_dct(Sw):
    n0, n1, n2, n2 = Sw.shape
    Shat = dct(Sw, axis=1,norm='ortho')
    Shat = dct(Shat, axis=0,norm='ortho')
    k=0
    condition = np.zeros((n0*n1))
    for i in range(n0):
        for j in range(n1):
            cond = np.linalg.cond(Shat[i, j, :, :])
            condition[k] = cond
            k+=1
    return condition
def tproddct4(A,B):
    n0,n1,n2,n3 = A.shape
    m0,m1,m2,m3 = B.shape
    if n0 != m0 and n1!=m1 and n3!=m2:
        print('warning, dimensions are not acceptable')
        return
    Ahat = dct(A, axis=1,norm='ortho')
    Ahat = dct(Ahat, axis=0,norm='ortho')
    Bhat = dct(B, axis=1,norm='ortho')
    Bhat = dct(Bhat, axis=0,norm='ortho')
    C = np.zeros((n0,n1,n2,m3))
    for i in range(n0):
        for j in range(n1):
            C[i,j,:,:] = Ahat[i,j,:,:]@Bhat[i,j,:,:]
    C1 = idct(C,axis=0,norm='ortho')
    C2 = idct(C1,axis=1,norm='ortho')
    return C2
def ttransdct4(A):
    n0, n1, n2, n3 = A.shape
    Ahat = dct(A, axis=1,norm='ortho')
    Ahat = dct(Ahat, axis=0,norm='ortho')
    B = np.zeros((n0,n1,n3,n2))
    for i in range(n0):
        for j in range(n1):
            B[i,j,:,:] = np.transpose((Ahat[i,j,:,:]))
    B1 = idct(B,axis=0,norm='ortho')
    B2 = idct(B1,axis=1,norm='ortho')
    return B2
def tinvdct4(A):
    n0, n1, n2, n3 = A.shape
    Ahat = dct(A, axis=1,norm='ortho')
    Ahat = dct(Ahat, axis=0,norm='ortho')
    B = np.zeros((n0,n1,n2,n3))
    for i in range(n0):
        for j in range(n1):
            B[i,j,:,:] = np.linalg.inv(Ahat[i,j,:,:])
    B1 = idct(B,axis=0,norm='ortho')
    B2 = idct(B1,axis=1,norm='ortho')
    return B2
def Class_scatters_dctcomp4(num_class,Tensor_train,y_train):
    n0,n1,n2,n3 = Tensor_train.shape
    mean_tensor_train = np.zeros((n0,n1,n2,num_class))
    Sw =  0
    Sb = 0
    a = np.zeros((n0,n1,n2,1))
    b = np.zeros((n0,n1, n2, 1))
    Mean_tensor = np.zeros((n0,n1, n2, 1))
    Mean_tensor[:,:,:,0] = (Tensor_train.sum(axis=3))/n3
    for i in range(num_class):
      Sa = 0
      occurrences = np.count_nonzero(y_train == i+1)
      idx = np.where(y_train==i+1)
      idx = idx[0]
      mean_tensor_train[:,:,:,i] = (Tensor_train[:,:,:,idx].sum(axis=3))/occurrences
      for j in idx:
          a[:,:,:,0] = Tensor_train[:,:,:,j]-mean_tensor_train[:,:,:,i]
          Sa = Sa + tproddct4(a,ttransdct4(a))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,:,0] = mean_tensor_train[:,:,:,i] - Mean_tensor[:,:,:,0]
        Sb = Sb + (tproddct4(b,ttransdct4(b)))*occurrences

    return Sw,Sb,mean_tensor_train
def preddct4(U_tr, U_tst, test_labels, train_labels):

    (k,l,m,n) = U_tr.shape

    (k1,l1,m1,n1) = U_tst.shape

    Ni = np.zeros((n,1))
    ClassTest = np.zeros((n1, 1),dtype=np.int32)
    for i in range(n1):
        for j in range(n):
            Ni[j, 0] = np.sqrt((np.sum((U_tst[:,:,:,i] - U_tr[:,:,:,j])**2)))
        idx = np.argmin(Ni)
        ClassTest[i, 0] = idx

    k =1
    test_pred = np.ones((n1,1))
    pRed = np.ones((n1,1))
    for i in range(n1):
        pRed[i] = train_labels[ClassTest[i]]
        if pRed[i] == test_labels[i]:
            test_pred[i] = k
            k = k + 1
        else:
            test_pred[i] = 0

    (a, b) = test_pred.shape
    accuracy = (np.amax(test_pred)*100)/(a)
    return test_pred, accuracy

from tensorly import unfold,fold
A = np.random.rand(10,10,10,10)
A_hat  = dct(A, axis=1,norm='ortho')
Ahat = dct(A_hat, axis=0,norm='ortho')

dct_matrix = dct(np.eye(10), axis=0,norm='ortho')
idct_matrix = idct(np.eye(10), axis=0,norm='ortho')

A_hat2 = fold(dct_matrix@unfold(A,1),1,[10,10,10,10])
A_hat3 = fold(dct_matrix@unfold(A_hat2,0),0,[10,10,10,10])

A_hat22 = fold(idct_matrix@unfold(A_hat3,1),1,[10,10,10,10])
A_hat33 = fold(idct_matrix@unfold(A_hat22,0),0,[10,10,10,10])

