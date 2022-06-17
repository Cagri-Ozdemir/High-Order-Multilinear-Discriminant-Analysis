import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
def tsvd4(A):
    n0,n1,n2,n3 = A.shape
    Ahat = np.fft.fft(A,axis=1)
    Ahat = np.fft.fft(Ahat,axis=0)
    U = np.zeros((n0,n1,n2,n2),dtype="complex")
    S = np.zeros((n0,n1,n2,n3),dtype="complex")
    V = np.zeros((n0,n1,n2,n2),dtype="complex")
    for i in range(n0):
        for j in range(n1):
            u,s,v = np.linalg.svd(Ahat[i,j,:,:],full_matrices=True)
            np.fill_diagonal(S[i,j, :, :], s)
            U[i,j,:,:] = u
            V[i,j,:,:] = (np.conj(v)).T
    U1 = np.fft.ifft(U,axis=0)
    U2 = np.fft.ifft(U1,axis=1)
    S1 = np.fft.ifft(S, axis=0)
    S2 = np.fft.ifft(S1, axis=1)
    V1 = np.fft.ifft(V, axis=0)
    V2 = np.fft.ifft(V1, axis=1)
    return U2,S2,V2
def teig4(A):
    n0, n1, n2, n3 = A.shape
    Ahat = np.fft.fft(A, axis=1)
    Ahat = np.fft.fft(Ahat, axis=0)
    U = np.zeros((n0, n1, n2, n2), dtype="complex")
    S = np.zeros((n0, n1, n2, n3), dtype="complex")
    for i in range(n0):
        for j in range(n1):
            s,u = np.linalg.eig(Ahat[i,j,:,:])
            idx = np.argsort(s)
            idx = idx[::-1][:n2]
            s = s[idx]
            u = u[:, idx]
            #s, u = linalg.cdf2rdf(s, u)
            #idx = np.argsort(s)
            #s = s[idx]
            #u = u[:, idx]
            np.fill_diagonal(S[i,j, :, :], s)
            U[i,j,:,:] = u
    U1 = np.fft.ifft(U, axis=0)
    U2 = np.fft.ifft(U1, axis=1)
    S1 = np.fft.ifft(S, axis=0)
    S2 = np.fft.ifft(S1, axis=1)
    return S2.real,U2.real
def tprod4(A,B):
    n0,n1,n2,n3 = A.shape
    m0,m1,m2,m3 = B.shape
    if n0 != m0 and n1!=m1 and n3!=m2:
        print('warning, dimensions are not acceptable')
        return
    Ahat = np.fft.fft(A, axis=1)
    Ahat = np.fft.fft(Ahat, axis=0)
    Bhat = np.fft.fft(B, axis=1)
    Bhat = np.fft.fft(Bhat, axis=0)
    C = np.zeros((n0,n1,n2,m3),dtype="complex")
    for i in range(n0):
        for j in range(n1):
            C[i,j,:,:] = Ahat[i,j,:,:]@Bhat[i,j,:,:]
    C1 = np.fft.ifft(C,axis=0)
    C2 = np.fft.ifft(C1,axis=1)
    return C2.real
def ttrans4(A):
    n0, n1, n2, n3 = A.shape
    Ahat = np.fft.fft(A, axis=1)
    Ahat = np.fft.fft(Ahat, axis=0)
    B = np.zeros((n0,n1,n3,n2),dtype="complex")
    for i in range(n0):
        for j in range(n1):
            B[i,j,:,:] = np.transpose(np.conj(Ahat[i,j,:,:]))
    B1 = np.fft.ifft(B,axis=0)
    B2 = np.fft.ifft(B1,axis=1)
    return B2.real
def tinv4(A):
    n0, n1, n2, n3 = A.shape
    Ahat = np.fft.fft(A, axis=1)
    Ahat = np.fft.fft(Ahat, axis=0)
    B = np.zeros((n0,n1,n2,n3),dtype="complex")
    for i in range(n0):
        for j in range(n1):
            B[i,j,:,:] = np.linalg.inv(Ahat[i,j,:,:])
    B1 = np.fft.ifft(B,axis=0)
    B2 = np.fft.ifft(B1,axis=1)
    return B2.real
def Class_scatters_dftcomp4(num_class,Tensor_train,y_train):
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
          Sa = Sa + tprod4(a,ttrans4(a))
      Sw = Sw+Sa
    for i in range(num_class):
        occurrences = np.count_nonzero(y_train == i + 1)
        b[:,:,:,0] = mean_tensor_train[:,:,:,i] - Mean_tensor[:,:,:,0]
        Sb = Sb + ((tprod4(b,ttrans4(b)))*occurrences)

    return Sw,Sb
def pred4(U_tr, U_tst, test_labels, train_labels):

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
    return test_pred, accuracy,pRed


def combined_classifier(u_row,u_clm,tst,trn,y_tst,y_trn):
    d0,d1,d2,d3 = tst.shape
    dd0,dd1,dd2,dd3 = trn.shape
    uuu = tprod4(u_clm, ttrans4(u_clm))
    uu = tprod4(u_row, ttrans4(u_row))
    row_projected = tprod4(ttrans4(u_row), trn)
    rp0,rp1,rp2,rp3 = row_projected.shape
    clm_projected = tprod4(ttrans4(u_clm), np.swapaxes(trn,1,2))
    cp0, cp1, cp2, cp3 = clm_projected.shape
    Predicted = np.zeros((d3))
    tst_slice = np.zeros((d0,d1,d2,1))
    row_projected_slice = np.zeros((rp0,rp1,rp2,1))
    clm_projected_slice = np.zeros((cp0, cp1, cp2, 1))
    winners = np.zeros((d3))
    for i in range(d3):
        tst_slice[:,:,:,0] = tst[:,:,:,i]
        d_1 = np.swapaxes(tst_slice, 1, 2) - (tprod4(uuu, np.swapaxes(tst_slice, 1, 2)))
        fd1 = np.sqrt(np.sum(d_1 ** 2)).real

        d_2 = tst_slice - (tprod4(uu, tst_slice))
        fd2 = np.sqrt(np.sum(d_2 ** 2)).real
        Ni = np.zeros((dd3))
        if fd1>fd2:
            winners[i] = 1
            projected_tensor = tprod4(ttrans4(u_row),tst_slice)
            for j in range(dd3):
                row_projected_slice[:,:,:,0] = row_projected[:, :, :, j]
                Ni[j] = np.sqrt((np.sum((projected_tensor - row_projected_slice) ** 2)))
            idx = np.argmin(Ni)
            Predicted[i] = y_trn[idx]
        else:
            winners[i] = 2
            projected_tensor = tprod4(ttrans4(u_clm), np.swapaxes(tst_slice, 1, 2))
            for j in range(dd3):
                clm_projected_slice[:,:,:,0] = clm_projected[:, :, :, j]
                Ni[j] = np.sqrt((np.sum((projected_tensor - clm_projected_slice) ** 2)))
            idx = np.argmin(Ni)
            Predicted[i] = y_trn[idx]
    C = Predicted == y_tst
    true_count = sum(C)
    accuracy = (true_count*100) / d3
    return accuracy,C,winners



def tprod5(A,B):
    n0,n1,n2,n3,n4 = A.shape
    m0,m1,m2,m3,m4 = B.shape
    if n0 != m0 and n1!=m1 and n2!=m2 and n4!=m3:
        print('warning, dimensions are not acceptable')
        return
    Ahat1 = np.fft.fft(A, axis=0)
    Ahat2 = np.fft.fft(Ahat1, axis=1)
    Ahat = np.fft.fft(Ahat2, axis=2)
    Bhat1 = np.fft.fft(B, axis=0)
    Bhat2 = np.fft.fft(Bhat1, axis=1)
    Bhat = np.fft.fft(Bhat2, axis=2)
    C = np.zeros((n0,n1,n2,n3,m4),dtype="complex")
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
              C[i,j,k,:,:] = Ahat[i,j,k,:,:]@Bhat[i,j,k,:,:]
    C1 = np.fft.ifft(C,axis=2)
    C2 = np.fft.ifft(C1,axis=1)
    C3 = np.fft.ifft(C2, axis=0)
    return C3.real

def updated_Sw_dft(Sw):
    n0, n1, n2, n2 = Sw.shape
    Shat = np.fft.fft(Sw, axis=1)
    Shat = np.fft.fft(Shat, axis=0)
    U = np.zeros((n0, n1, n2, n2),dtype="complex")
    S = np.zeros((n0, n1, n2, n2),dtype="complex")
    k=0
    for i in range(n0):
        for j in range(n1):
            cond = np.linalg.cond(Shat[i, j, :, :])
            if 1 / cond <= 1.e-5:
                k+=1
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
    U1 = np.fft.ifft(U, axis=0)
    U2 = np.fft.ifft(U1, axis=1)
    S1 = np.fft.ifft(S, axis=0)
    S2 = np.fft.ifft(S1, axis=1)
    ww1 = tprod4(U2, S2)
    Sww1 = tprod4(ww1, ttrans4(U2))
    return Sww1.real,k

def condition_dft(Sw):
    n0, n1, n2, n2 = Sw.shape
    Shat = np.fft.fft(Sw, axis=1)
    Shat = np.fft.fft(Shat, axis=0)
    condition = np.zeros((n0*n1))
    k=0
    for i in range(n0):
        for j in range(n1):
            cond = np.linalg.cond(Shat[i, j, :, :])
            condition[k] = cond
            k +=1
    return condition
#S,U=teig4(A)
#F1 = tprod4(U,S)
#F2 = (tprod4(F1,tinv4(U))).real

#Ahat = np.fft.fft(Tensor_train, axis=1)
#Ahat = np.fft.fft(Ahat, axis=0).real
