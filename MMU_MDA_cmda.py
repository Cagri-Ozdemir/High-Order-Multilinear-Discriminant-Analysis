import scipy.io
import numpy as np
from CMDA import Class_scatters_cmda
from tensorly import unfold,fold
from FFT import pred4
irisdf = scipy.io.loadmat('IRIS.mat')
iris = irisdf['AD']
label = np.zeros((225))
d0,d1,d2,d3 = iris.shape
k = 0
for i in range(45):
    for ii in range(5):
        label[k] = i+1
        k+=1

import pickle
K_fold_dic = pickle.load(open("k_fold_dic_MMU.pkl","rb"))
##########
###
dim1= 10
dim2 = 10
tmax=100
accuracy_cmda= np.zeros((5))
##########
for i in range(5):
    s = K_fold_dic[i]
    Train_set = iris[:,:,:,s[0]]
    Test_set = iris[:,:,:,s[1]]
    Ytrn = label[s[0]]
    Ytst = label[s[1]]
    trn = s[0].shape[0]
    tst = s[1].shape[0]
    U0,U1,U2 = Class_scatters_cmda(Train_set,Ytrn,d0,d1,d2,100)
    ###Projected tensors (train)
    tr0 = unfold(Train_set,0)
    P0 = fold(U0.T@tr0,0,[d0,d1,d2,trn])

    tr1 = unfold(P0,1)
    P1 = fold(U1[:,0:dim1].T@tr1,1,[d0,dim1,d2,trn])

    tr2 = unfold(P1,2)
    P2 = fold(U2[:,0:dim2].T@tr2,2,[d0,dim1,dim2,trn])

    ###Projected tensors (test)
    tst0 = unfold(Test_set,0)
    PP0 = fold(U0.T@tst0,0,[d0,d1,d2,tst])

    tst1 = unfold(PP0,1)
    PP1 = fold(U1[:,0:dim1].T@tst1,1,[d0,dim1,d2,tst])

    tst2 = unfold(PP1,2)
    PP2 = fold(U2[:,0:dim2].T@tst2,2,[d0,dim1,dim2,tst])
    ##nearest neighbor search
    pred1,ac1,pRed1 = pred4(P2,PP2,Ytst,Ytrn)
    accuracy_cmda[i] = ac1

