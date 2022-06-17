import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from FFT import pred4
from DWT import teigdwt4,ttransdwt4,tinvdwt4,Class_scatters_dwtcomp4,tproddwt4,updated_Sw_dwt

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
accuracy_dwt_clm = np.zeros((5))
accuracy_dwt_row = np.zeros((5))
accuracy_dwt_combined = np.zeros((5))

eig = 10
for i in range(5):
    s = K_fold_dic[i]
    Train_set = iris[:,:,:,s[0]]
    Test_set = iris[:,:,:,s[1]]
    Ytrn = label[s[0]]
    Ytst = label[s[1]]

    ###################SET1#########
    #######################################################~FOURIER TRANSFORM~capitalize column pixels#########################################
    Sw1, Sb1 = Class_scatters_dwtcomp4(45, Train_set, Ytrn)
    S1 = tproddwt4(tinvdwt4(Sw1), Sb1)
    SS1, UU1 = teigdwt4(S1)
    u1 = UU1[:, :, :, 0:eig]
    pro_df_trn1 = tproddwt4(ttransdwt4(u1), Train_set)
    pro_df_tst1 = tproddwt4(ttransdwt4(u1), Test_set)
    pre1, accuracy1, pRed1 = pred4(pro_df_trn1, pro_df_tst1, Ytst, Ytrn)
    accuracy_dwt_clm[i] = accuracy1
    print("ac1:", accuracy1)
    ############
    Sww1, k = updated_Sw_dwt(Sw1)
    S1_u = tproddwt4(tinvdwt4(Sww1), Sb1)
    # S1_u = updated_Sw(S1_u)
    SS1_u, UU1_u = teigdwt4(S1_u)
    u1_u = UU1_u[:, :, :, 0:eig]
    pro_df_trn1_u = tproddwt4(ttransdwt4(u1_u), Train_set)
    pro_df_tst1_u = tproddwt4(ttransdwt4(u1_u), Test_set)
    pre1_up, accuracy1_up, pRed1_up = pred4(pro_df_trn1_u, pro_df_tst1_u, Ytst, Ytrn)
    accuracy_dwt_row[i] = accuracy1_up
    print("ac1_up", accuracy1_up)

print("dft_row_mean",np.mean(accuracy_dwt_row))
print("dft_row_std",np.std(accuracy_dwt_row))

print("dft_clm_mean",np.mean(accuracy_dwt_clm))
print("dft_clm_std",np.std(accuracy_dwt_clm))
